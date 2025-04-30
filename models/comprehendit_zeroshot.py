from transformers import pipeline
import pandas as pd
from datasets import Dataset
import logging
import torch
import os

import data_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_comprehendit_pipeline(device_num=-1):
    ### Loads the ComprehendIt zero-shot classification pipeline
    model_name = "knowledgator/comprehend_it-base"
    logging.info(f"Loading ComprehendIt pipeline: {model_name}")
    try:
        target_device_idx = -1 # Default CPU
        if device_num >= 0 and torch.cuda.is_available(): target_device_idx = device_num
        elif device_num >= 0 and not torch.cuda.is_available(): logging.warning(f"GPU {device_num} requested but CUDA not available. Using CPU.")
        classifier = pipeline("zero-shot-classification", model=model_name, device=target_device_idx)
        logging.info(f"ComprehendIt pipeline loaded. Device: {classifier.device}")
        return classifier
    except Exception as e:
        logging.error(f"Failed to load ComprehendIt pipeline {model_name}: {e}", exc_info=True)
        raise

def format_comprehendit_sequence(question, option_letter, option_text):
    ### Formats the input sequence for the classification task
    return f"Question: {question}\nIs option {option_letter}: '{option_text}' the correct answer?"


def run_comprehendit_inference(classifier, test_file, batch_size=16, subject_match=None):
    ### Runs inference using the ComprehendIt pipeline. ###
    logging.info(f"Loading test data: {test_file}")
    test_kwargs = {'subject_match': subject_match} if subject_match else {}
    test_df = data_utils.load_and_transform_json(test_file, **test_kwargs)

    if test_df.empty:
        logging.error("Test data is empty. Cannot run inference.")
        return None, None, None, None

    ### Prepare Ground Truth and Run Inference ###
    num_to_letter = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
    ground_truth = [num_to_letter.get(int(gt), str(gt)) for gt in test_df['cop'].tolist()]
    letter_to_col = {'A': 'opa', 'B': 'opb', 'C': 'opc', 'D': 'opd'}
    ground_truth_text = []
    # Get text for ground truth letters
    for i, gt_letter in enumerate(ground_truth):
        col_name = letter_to_col.get(gt_letter)
        if col_name and col_name in test_df.columns: ground_truth_text.append(str(test_df.iloc[i][col_name]))
        else: ground_truth_text.append("")

    predictions, predictions_text = [], []
    possible_labels = ["Correct", "Incorrect"] # Labels for the classifier
    logging.info(f"Running ComprehendIt inference on {len(test_df)} samples...")

    # Process row by row, classifying all options for a question together
    for index, row in test_df.iterrows():
        option_scores = {}
        options_map = {'A': row['opa'], 'B': row['opb'], 'C': row['opc'], 'D': row['opd']}
        sequences_to_classify, letters_for_sequences = [], []

        # Prepare sequences for all valid options in this row
        for letter, text in options_map.items():
            if text and pd.notna(text):
                sequences_to_classify.append(format_comprehendit_sequence(row['question'], letter, text))
                letters_for_sequences.append(letter)

        if not sequences_to_classify: # Handle rows with no valid options
            predictions.append('?')
            predictions_text.append("")
            continue

        # Run classifier on all sequences for this question
        try:
            results = classifier(sequences_to_classify, possible_labels, multi_label=False)
            if not isinstance(results, list): results = [results] # Ensure list format

            # Extract scores for the 'Correct' label
            for i, res in enumerate(results):
                correct_score = 0.0
                current_letter = letters_for_sequences[i]
                try:
                    for label_idx, label in enumerate(res['labels']):
                        if label == possible_labels[0]: # Check for "Correct" label
                            correct_score = res['scores'][label_idx]
                            break
                except Exception as res_err: # Catch potential issues with result format
                     logging.warning(f"Could not parse score for row {index}, option {current_letter}. Result: {res}. Error: {res_err}")
                option_scores[current_letter] = correct_score

        except Exception as e:
            logging.warning(f"Pipeline error processing row {index}: {e}. Sequences: {len(sequences_to_classify)}")
            for letter in letters_for_sequences: option_scores[letter] = -1.0 # Penalize on error

        # Determine prediction based on highest score
        if option_scores: predicted_letter = max(option_scores, key=option_scores.get)
        else: predicted_letter = '?' # Default if scoring failed

        predictions.append(predicted_letter)
        # Get corresponding prediction text
        pred_text = options_map.get(predicted_letter, "")
        predictions_text.append(pred_text)

        if (index + 1) % 100 == 0: logging.info(f"Processed {index + 1} / {len(test_df)} samples")

    logging.info("ComprehendIt inference complete.")
    return predictions, ground_truth, predictions_text, ground_truth_text