import torch
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer, Seq2SeqTrainer, Seq2SeqTrainingArguments # type: ignore
from datasets import Dataset
import logging
import os

from .flan_tokenize import preprocess as flan_preprocess_external
from utilities.data_utils import load_and_transform_json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_NAME = "google/flan-t5-small"
flan_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_data_internal(data):
    
    inputs = [f"Question: {q} Options: A. {a}, B. {b}, C. {c}, D. {d} Answer:"
              for q, a, b, c, d in zip(data["question"], data["opa"], data["opb"], 
              data["opc"], data["opd"])]
    targets = [str(item) for item in data["cop"]]
    num_to_letter = {1: "A", 2: "B", 3: "C", 4: "D"}
    targets = [num_to_letter.get(int(t), str(t)) for t in targets] 

    model_inputs = flan_tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = flan_tokenizer(targets, max_length=8, truncation=True, padding="max_length")
    labels_input_ids = labels["input_ids"]

    processed_labels = [
        [(1 if l != flan_tokenizer.pad_token_id else -100) for l in label]
        for label in labels
    ]

    model_inputs["labels"] = processed_labels
    return model_inputs

def train_flan_t5(train_file, eval_file, model_output_dir, num_epochs=3, batch_size=8, learning_rate=2e-5, subject_match=None):
    logging.info(f"Starting FLAN-T5 fine-tuning. Training Data: {train_file}, Evaluation Data: {eval_file}")
    train_kwargs = {'subject_match': subject_match} if subject_match else {}
    eval_kwargs = {'subject_match': subject_match} if subject_match else {}

    train_df = load_and_transform_json(train_file, **train_kwargs)
    eval_df = load_and_transform_json(eval_file, **eval_kwargs)

    if train_df.empty or eval_df.empty:
        logging.error("Training or evaluation dataframe is empty")
        return
    
    logging.info(f"Loaded {len(train_df)} training samples and {len(eval_df)} evaluation samples.")
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    logging.info("Tokenizing and preprocessing datasets...")
    tokenized_train_dataset = train_dataset.map(preprocess_data_internal, batched=True, remove_columns=train_df.column_names)
    tokenized_eval_dataset = train_dataset.map(preprocess_data_internal, batched=True, remove_columns=eval_df.column_names)
    logging.info("Preprocessing complete.")

    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    except Exception as e:
        logging.error(f"Failed to load base model {MODEL_NAME}: {e}")
        return

    training_args = Seq2SeqTrainingArguments(
        output_dir=model_output_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_dir=os.path.join(model_output_dir, 'logs'),
        logging_steps=100,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=flan_tokenizer,
    )

    logging.info("Starting training...")
    try:
        train_result = trainer.train()
        logging.info("Training finished.")
        logging.info(f"Training results: {train_result}")

        final_model_path = os.path.join(model_output_dir, "final_model")
        trainer.save_model(final_model_path)
        flan_tokenizer.save_pretrained(final_model_path) # Saving the tokenizer alongside the trained model
        logging.info(f"Best model saved to {final_model_path}")
    except Exception as e:
        logging.error(f"An error has occurred during training: {e}", exc_info=True)
    

def run_flan_inference(model_path, test_file, batch_size=16, subject_match=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Runs inference using the fine-tuned FLAN-T5 model.
    logging.info(f"Loading fine-tuned FLAN model from: {model_path}")
    try:
        # Load the model and tokenizer specific to this run
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path) # Load the tokenizer saved with the model
        model.eval()
    except Exception as e:
        logging.error(f"Failed to load model/tokenizer from {model_path}: {e}")
        return None, None, None, None # Return four Nones

    logging.info(f"Loading test data from: {test_file}")
    test_kwargs = {'subject_match': subject_match} if subject_match else {}
    test_df = load_and_transform_json(test_file, **test_kwargs)
    if test_df.empty:
        logging.error("Test dataframe is empty.")
        return None, None, None, None # Return four Nones

    num_to_letter = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
    ground_truth = [num_to_letter.get(int(gt), str(gt)) for gt in test_df['cop'].tolist()]
    letter_to_col = {'A': 'opa', 'B': 'opb', 'C': 'opc', 'D': 'opd'}

    # Pre-calculate ground truth text to avoid any repeated lookups in loop
    ground_truth_text = []
    for i, gt_letter in enumerate(ground_truth):
        col_name = letter_to_col.get(gt_letter)
        if col_name and col_name in test_df.columns:
            ground_truth_text.append(str(test_df.iloc[i][col_name]))
        else:
            ground_truth_text.append("") # Append an empty string if mapping fails
            logging.warning(f"Could not map ground truth letter '{gt_letter}' to option column for row {i}")

    test_dataset = Dataset.from_pandas(test_df)
    predictions = []
    predictions_text = []

    logging.info(f"Running inference on {len(test_dataset)} samples...")
    with torch.no_grad():
        for i in range(0, len(test_dataset), batch_size):
            batch_data = test_dataset[i : i + batch_size]
            inputs = [
                f"Question: {q} Options: A. {a}, B. {b}, C. {c}, D. {d} Answer:"
                for q, a, b, c, d in zip(
                    batch_data["question"], batch_data["opa"], batch_data["opb"],
                    batch_data["opc"], batch_data["opd"])
            ]
            input_tokens = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            output_tokens = model.generate(**input_tokens, max_new_tokens=8)
            batch_preds = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

            for idx, pred_letter in enumerate(batch_preds):
                clean_pred = pred_letter.strip().upper()
                final_pred_letter = '?'
                if clean_pred and clean_pred[0] in ['A', 'B', 'C', 'D']:
                    final_pred_letter = clean_pred[0]
                predictions.append(final_pred_letter)

                # Get prediction text safely
                pred_col_name = letter_to_col.get(final_pred_letter)
                current_row_index = test_df.index[i+idx] # Get the original DataFrame index
                if pred_col_name and pred_col_name in test_df.columns:
                     predictions_text.append(str(test_df.loc[current_row_index, pred_col_name]))
                else:
                     predictions_text.append("") # Append empty string if mapping fails


            if (i // batch_size + 1) % 10 == 0:
                 logging.info(f"Processed {i + len(batch_data)} / {len(test_dataset)} samples")

    logging.info("Inference complete.")
    return predictions, ground_truth, predictions_text, ground_truth_text # Return four lists