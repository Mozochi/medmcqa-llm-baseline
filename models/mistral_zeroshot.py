import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
from datasets import Dataset
import logging
import re
import os

import data_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_mistral_model(model_id="mistralai/Mistral-7B-Instruct-v0.1", use_4bit=True):
    ### Loads the Mistral model, optionally with 4-bit quantization.
    logging.info(f"Loading Mistral model: {model_id}")
    bnb_config = None
    load_kwargs = {"trust_remote_code": True}

    if use_4bit and torch.cuda.is_available():
        logging.info("Using 4-bit quantization.")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        load_kwargs["quantization_config"] = bnb_config
    elif use_4bit and not torch.cuda.is_available():
        logging.warning("CUDA not available, cannot use 4-bit. Loading in default precision.")

    if torch.cuda.is_available():
         load_kwargs["device_map"] = "auto"
    else:
         load_kwargs["device_map"] = "cpu"
         logging.info("CUDA not available. Mapping model to CPU.")

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token # Set pad token
        model.eval()
        logging.info("Mistral model and tokenizer loaded.")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Failed to load Mistral model {model_id}: {e}", exc_info=True)
        raise

def format_mistral_prompt(question, option_a, option_b, option_c, option_d):
    ### Creates the instruction prompt for Mistral. ###
    prompt = f"""<s>[INST] You are an expert medical AI assistant. Answer the following multiple-choice question by choosing the single best option.
Question: {question}
Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

Provide only the letter of the correct option (A, B, C, or D). [/INST]"""
    return prompt

def parse_mistral_output(generated_text):
    ### Extracts the predicted letter (A, B, C, D) from the model's response

    cleaned_text = generated_text.split("[/INST]")[-1].strip()
    match = re.search(r'\b([A-D])\b', cleaned_text)
    if match: return match.group(1)
    if cleaned_text and cleaned_text[0] in ['A', 'B', 'C', 'D']: return cleaned_text[0]
    logging.warning(f"Could not parse valid option from Mistral output: '{cleaned_text}'")
    return '?'


def run_mistral_inference(model, tokenizer, test_file, batch_size=4, subject_match=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    ### Runs zero-shot inference with the Mistral model.
    logging.info(f"Loading test data: {test_file}")
    test_kwargs = {'subject_match': subject_match} if subject_match else {}
    # Use imported data_utils
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

    test_dataset = Dataset.from_pandas(test_df)
    predictions, predictions_text = [], []
    model_device = model.device # Use the device the model is actually on
    logging.info(f"Running Mistral inference on {len(test_dataset)} samples (device: {model_device})...")

    with torch.no_grad():
        for i in range(0, len(test_dataset), batch_size):
            batch_data = test_dataset[i : i + batch_size]
            # Create prompts for the batch
            prompts = [format_mistral_prompt(q, a, b, c, d) for q, a, b, c, d in zip(batch_data["question"], batch_data["opa"], batch_data["opb"], batch_data["opc"], batch_data["opd"])]
            # Tokenize and send to model's device
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model_device)
            # Generate predictions
            generated_ids = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id, do_sample=False)
            # Decode generated text
            batch_outputs_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # Process predictions for this batch
            for idx, full_output in enumerate(batch_outputs_text):
                pred_letter = parse_mistral_output(full_output)
                predictions.append(pred_letter)
                # Get corresponding prediction text
                pred_col_name = letter_to_col.get(pred_letter)
                current_row_index = test_df.index[i+idx]
                if pred_col_name and pred_col_name in test_df.columns: predictions_text.append(str(test_df.loc[current_row_index, pred_col_name]))
                else: predictions_text.append("")

            if (i // batch_size + 1) % 5 == 0: logging.info(f"Processed {i + len(batch_data)} / {len(test_dataset)} samples")

    logging.info("Mistral inference complete.")
    return predictions, ground_truth, predictions_text, ground_truth_text