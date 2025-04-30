import gradio as gr
import pandas as pd
import numpy as np
from datasets import Dataset
import time
import os
import torch
import logging


try:

    import data_utils
    import evaluation 

    from models.flan_finetune import run_flan_inference as run_flan_inference_batch # Rename batch if needed
    from models.mistral_zeroshot import load_mistral_model, run_mistral_inference as run_mistral_inference_batch, format_mistral_prompt, parse_mistral_output
    from models.comprehendit_zeroshot import load_comprehendit_pipeline, run_comprehendit_inference as run_comprehendit_inference_batch, format_comprehendit_sequence

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    print("[gui.py] Successfully imported project modules from root.")
except ImportError as e:
    print(f"!!! [gui.py] FAILED to import project modules from root: {e}")
    data_utils = None 


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

### Global State & Configuration ###
LOADED_MODELS = {"flan": None, "mistral": None, "comprehendit": None} # Cache loaded models
DEV_DF = None # Cache dev data
DEFAULT_OUTPUT_DIR = "./mcqa_output"
DEFAULT_FLAN_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "flan_t5_finetuned", "final_model")
DEFAULT_MISTRAL_ID = "mistralai/Mistral-7B-Instruct-v0.1"


### GUI Helper Functions ###

def load_dev_data(data_dir):
    ### Loads and caches dev data for random examples.
    global DEV_DF
    if DEV_DF is not None: return DEV_DF
    if not data_utils: 
        logging.error("data_utils module unavailable. Cannot load dev data.")
        return None
    dev_file = os.path.join(data_dir, 'dev.json')
    try:
        DEV_DF = data_utils.json_to_dataframe(dev_file) 
        logging.info(f"Loaded {len(DEV_DF)} dev samples from {dev_file}")
        return DEV_DF
    # ... (error handling as before) ...
    except FileNotFoundError: logging.error(f"Dev file not found: {dev_file}"); DEV_DF = pd.DataFrame(); return None
    except Exception as e: logging.error(f"Error loading dev data: {e}"); DEV_DF = pd.DataFrame(); return None


def get_random_example(data_dir):
    ### Fetches a random example from the loaded dev data
    df = load_dev_data(data_dir)
    if df is not None and not df.empty:
         random_row = df.sample(1).iloc[0]
         num_to_letter = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
         # Safely get values using .get() with defaults
         q = random_row.get('question', 'N/A')
         opa = str(random_row.get('opa', ''))
         opb = str(random_row.get('opb', ''))
         opc = str(random_row.get('opc', ''))
         opd = str(random_row.get('opd', ''))
         cop_val = random_row.get('cop', 0) # Default to 0 if missing
         try: correct_letter = num_to_letter.get(int(cop_val), '?')
         except (ValueError, TypeError): correct_letter = '?'

         # Get text of correct answer
         correct_text = "N/A"
         if correct_letter != '?':
             correct_col_name = {'A': 'opa', 'B': 'opb', 'C': 'opc', 'D': 'opd'}.get(correct_letter)
             if correct_col_name: correct_text = str(random_row.get(correct_col_name, 'N/A'))

         return q, opa, opb, opc, opd, f"Correct Answer: {correct_letter} ({correct_text})"
    else:
        return "Error loading data", "", "", "", "", ""


def load_results_from_csv(output_dir):
    ### Loads final evaluation metrics from the results CSV.
    results_file = os.path.join(output_dir, "evaluation_results.csv")
    try:
        df = pd.read_csv(results_file, index_col=0)
        return df.round(4)
    except FileNotFoundError: return pd.DataFrame({"Status": [f"Results file not found: {results_file}."]})
    except Exception as e: return pd.DataFrame({"Status": [f"Error loading results: {e}"]})


def get_model(model_type, flan_path=None, mistral_id=None, use_4bit=True):
    ### Loads and caches models when first needed by the GUI ###
    global LOADED_MODELS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_index = 0 if torch.cuda.is_available() else -1

    if model_type == "flan":
        if LOADED_MODELS["flan"] is None:
            path_to_load = flan_path or DEFAULT_FLAN_PATH

            abs_path_to_load = os.path.abspath(path_to_load)

            logging.info(f"Loading FLAN model for GUI from (absolute): {abs_path_to_load}")

            # Check if the directory exists before attempting to load
            if not os.path.isdir(abs_path_to_load):
                 logging.error(f"FLAN model directory not found: {abs_path_to_load}")
                 LOADED_MODELS["flan"] = "error" # Mark as error
                 return None # Explicitly return None if path doesn't exist

            try:

                model = AutoModelForSeq2SeqLM.from_pretrained(abs_path_to_load).to(device)
                tokenizer = AutoTokenizer.from_pretrained(abs_path_to_load)

                model.eval()
                LOADED_MODELS["flan"] = (model, tokenizer)
                logging.info("FLAN model loaded.")
            except Exception as e:
                # Log the specific error for better debugging
                logging.error(f"Failed to load FLAN model from {abs_path_to_load}: {e}", exc_info=True)
                LOADED_MODELS["flan"] = "error" # Mark as error
        # Return the cached resource or None if error occurred
        return LOADED_MODELS["flan"] if LOADED_MODELS["flan"] != "error" else None


    elif model_type == "mistral":
        if LOADED_MODELS["mistral"] is None:
             model_id_to_load = mistral_id or DEFAULT_MISTRAL_ID
             logging.info(f"Loading Mistral model for GUI: {model_id_to_load}")
             try:
                 model, tokenizer = load_mistral_model(model_id_to_load, use_4bit=use_4bit) # Use imported loader
                 LOADED_MODELS["mistral"] = (model, tokenizer)
                 logging.info("Mistral model loaded.")
             except Exception as e: logging.error(f"Failed to load Mistral model {model_id_to_load}: {e}", exc_info=True); LOADED_MODELS["mistral"] = "error"
        return LOADED_MODELS["mistral"] if LOADED_MODELS["mistral"] != "error" else None

    elif model_type == "comprehendit":
         if LOADED_MODELS["comprehendit"] is None:
             logging.info("Loading ComprehendIt pipeline for GUI...")
             try:
                 pipeline_obj = load_comprehendit_pipeline(device_num=gpu_index) # Use imported loader
                 LOADED_MODELS["comprehendit"] = pipeline_obj
                 logging.info("ComprehendIt pipeline loaded.")
             except Exception as e: logging.error(f"Failed to load ComprehendIt pipeline: {e}", exc_info=True); LOADED_MODELS["comprehendit"] = "error"
         return LOADED_MODELS["comprehendit"] if LOADED_MODELS["comprehendit"] != "error" else None
    else:
        logging.error(f"Unknown model type requested: {model_type}")
        return None


def predict_single(model_type, question, opt_a, opt_b, opt_c, opt_d, flan_path=None, mistral_id=None, use_4bit=True):
    ### Runs prediction for one example using the selected model

    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    options_map = {'A': opt_a, 'B': opt_b, 'C': opt_c, 'D': opt_d}
    model_resource = get_model(model_type, flan_path, mistral_id, use_4bit)
    if model_resource is None or model_resource == "error": return f"Error loading {model_type} model.", "0.0 sec"

    prediction = "?"
    duration = 0.0
    try:
        if model_type == "flan":
            model, tokenizer = model_resource
            input_text = f"Question: {question} Options: A. {opt_a}, B. {opt_b}, C. {opt_c}, D. {opt_d} Answer:"
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
            with torch.no_grad(): output_tokens = model.generate(**inputs, max_new_tokens=8)
            pred_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True).strip().upper()
            if pred_text and pred_text[0] in ['A', 'B', 'C', 'D']: prediction = pred_text[0]

        elif model_type == "mistral":
            model, tokenizer = model_resource
            model_device = model.device
            prompt = format_mistral_prompt(question, opt_a, opt_b, opt_c, opt_d) 
            inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(model_device)
            with torch.no_grad(): generated_ids = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id, do_sample=False)
            output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            prediction = parse_mistral_output(output_text) 

        elif model_type == "comprehendit":
            classifier = model_resource
            possible_labels = ["Correct", "Incorrect"]
            option_scores = {}
            sequences, letters = [], []
            for letter, text in options_map.items():
                 if text and pd.notna(text):
                     sequences.append(format_comprehendit_sequence(question, letter, text)) 
                     letters.append(letter)
            if sequences:
                 results = classifier(sequences, possible_labels, multi_label=False)
                 if not isinstance(results, list): results = [results]
                 for i, res in enumerate(results):
                    correct_score = 0.0; current_letter = letters[i]
                    try:
                         for label_idx, label in enumerate(res['labels']):
                             if label == possible_labels[0]: correct_score = res['scores'][label_idx]; break
                    except Exception: pass
                    option_scores[current_letter] = correct_score
            if option_scores: prediction = max(option_scores, key=option_scores.get)

    except Exception as e:
        logging.error(f"Error during {model_type} single prediction: {e}", exc_info=True)
        duration = time.time() - start_time
        return f"Prediction error: {str(e)[:100]}...", f"{duration:.3f} sec"

    end_time = time.time()
    duration = end_time - start_time
    pred_text_display = options_map.get(prediction, "N/A")
    return f"{prediction} ({pred_text_display})", f"{duration:.3f} sec"


### Create Gradio Interface ###

def create_gui(output_dir=DEFAULT_OUTPUT_DIR, data_dir='./data/MedMCQA', flan_path=DEFAULT_FLAN_PATH, mistral_id=DEFAULT_MISTRAL_ID, mistral_4bit=True):
    ### Builds the Gradio application UI

    with gr.Blocks(theme=gr.themes.Soft()) as interface:
        gr.Markdown("MCQA NLP")

        with gr.Tab("Summary Results"):
            gr.Markdown("View evaluation metrics loaded from `evaluation_results.csv` (generated by `evaluate_pipeline.py`).")
            load_button = gr.Button("Load Results")
            results_output = gr.DataFrame(label="Evaluation Metrics", wrap=True)
            load_button.click(lambda: load_results_from_csv(output_dir), [], [results_output])

        with gr.Tab("Test Single Example"):
            gr.Markdown("Run inference on a single example with a selected model.")
            with gr.Row():
                model_choice = gr.Dropdown(["flan", "mistral", "comprehendit"], label="Choose Model", value="mistral")
                load_random_button = gr.Button("Load Random Example")
            with gr.Column():
                q_input = gr.Textbox(label="Question", lines=3, placeholder="Enter question or load random")
                a_input = gr.Textbox(label="Option A")
                b_input = gr.Textbox(label="Option B")
                c_input = gr.Textbox(label="Option C")
                d_input = gr.Textbox(label="Option D")
                correct_answer_display = gr.Textbox(label="Correct Answer (if loaded)", interactive=False)
            predict_button = gr.Button("Run Prediction")
            with gr.Row():
                prediction_output = gr.Textbox(label="Model Prediction")
                time_output = gr.Textbox(label="Inference Time")

            ### Connect Buttons to Functions ###
            load_random_button.click(lambda: get_random_example(data_dir), [],
                                     [q_input, a_input, b_input, c_input, d_input, correct_answer_display])

            predict_button.click(predict_single,
                                 inputs=[model_choice, q_input, a_input, b_input, c_input, d_input,
                                         gr.State(flan_path), gr.State(mistral_id), gr.State(mistral_4bit)],
                                 outputs=[prediction_output, time_output])

    return interface