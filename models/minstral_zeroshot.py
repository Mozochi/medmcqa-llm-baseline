import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig # type: ignore
import pandas as pd
from datasets import Dataset
import logging
import re
import os

from utilities.data_utils import load_and_transform_json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_mistral_model(model_id="mistralai/Mistral-7B-Instruct-v0.1", use_4bit=True):
    # Loads the Mistral model and tokenizer, potentially with 4-bit quantization
    logging.info(f"Loading Mistral model: {model_id}")
    bnb_config = None
    if use_4bit:
        # Check if CUDA is available for 4-bit
        if not torch.cuda.is_available():
            logging.warning("CUDA not available, cannot use 4-bit quantization. Loading in default precision.")
            use_4bit = False # Override flag
        else:
            logging.info("Using 4-bit quantization.")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

    # Determine device map strategy
    if torch.cuda.is_available():
        device_map = "auto"
    else:
        device_map = "cpu"
        logging.info("CUDA not available. Mapping model to CPU.")


    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config if use_4bit else None, # Only pass config if using 4-bit
            device_map=device_map,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        logging.info("Mistral model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Failed to load Mistral model {model_id}: {e}", exc_info=True)
        raise

def format_mistral_prompt(question, option_a, option_b, option_c, option_d):
    # Creates a formatted prompt for Mistral
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
    # Extracts the predicted option letter (A, B, C, D) from Mistral's output
    cleaned_text = generated_text.split("[/INST]")[-1].strip()
    match = re.search(r'\b([A-D])\b', cleaned_text)
    if match:
        return match.group(1)
    if cleaned_text and cleaned_text[0] in ['A', 'B', 'C', 'D']:
         return cleaned_text[0]
    logging.warning(f"Could not parse valid option from Mistral output: '{cleaned_text}'")
    return '?'


def run_mistral_inference(model, tokenizer, test_file, batch_size=4, subject_match=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Runs zero-shot inference using the loaded Mistral model.
    logging.info(f"Loading test data from: {test_file}")
    test_kwargs = {'subject_match': subject_match} if subject_match else {}
    test_df = load_and_transform_json(test_file, **test_kwargs)
    if test_df.empty:
        logging.error("Test dataframe is empty.")
        return None, None, None, None # Return four Nones

    num_to_letter = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
    ground_truth = [num_to_letter.get(int(gt), str(gt)) for gt in test_df['cop'].tolist()]
    letter_to_col = {'A': 'opa', 'B': 'opb', 'C': 'opc', 'D': 'opd'}

    # Pre-calculate ground truth
    ground_truth_text = []
    for i, gt_letter in enumerate(ground_truth):
        col_name = letter_to_col.get(gt_letter)
        if col_name and col_name in test_df.columns:
            ground_truth_text.append(str(test_df.iloc[i][col_name]))
        else:
            ground_truth_text.append("")

    test_dataset = Dataset.from_pandas(test_df)
    predictions = []
    predictions_text = []

    logging.info(f"Running Mistral inference on {len(test_dataset)} samples...")
    # Use the models device determined during loading
    model_device = model.device if hasattr(model, 'device') else device # Fallback

    with torch.no_grad():
        for i in range(0, len(test_dataset), batch_size):
            batch_data = test_dataset[i : i + batch_size]
            prompts = [
                format_mistral_prompt(q, a, b, c, d)
                for q, a, b, c, d in zip(
                    batch_data["question"], batch_data["opa"], batch_data["opb"],
                    batch_data["opc"], batch_data["opd"])
            ]

            # Tokenize and send to the correct device
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model_device)

            generated_ids = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id, do_sample=False)
            batch_outputs_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for idx, full_output in enumerate(batch_outputs_text):
                pred_letter = parse_mistral_output(full_output)
                predictions.append(pred_letter)

                # Get prediction text 
                pred_col_name = letter_to_col.get(pred_letter)
                current_row_index = test_df.index[i+idx]
                if pred_col_name and pred_col_name in test_df.columns:
                     predictions_text.append(str(test_df.loc[current_row_index, pred_col_name]))
                else:
                     predictions_text.append("")


            if (i // batch_size + 1) % 5 == 0:
                 logging.info(f"Processed {i + len(batch_data)} / {len(test_dataset)} samples")

    logging.info("Mistral inference complete.")
    return predictions, ground_truth, predictions_text, ground_truth_text # Return four lists