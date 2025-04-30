import argparse
import pandas as pd
import logging
import os
import torch

### Project Imports ###
import data_utils
import evaluation
from models.flan_finetune import train_flan_t5, run_flan_inference
from models.mistral_zeroshot import load_mistral_model, run_mistral_inference
from models.comprehendit_zeroshot import load_comprehendit_pipeline, run_comprehendit_inference

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

### Setup Computation Device ###
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GPU_INDEX = 0 if torch.cuda.is_available() else -1 # For pipeline device index
logging.info(f"Using device: {DEVICE}")

### Wrap Inference Functions for Timing ###
# Apply the time_it decorator from evaluation module
timed_run_flan_inference = evaluation.time_it(run_flan_inference)
timed_run_mistral_inference = evaluation.time_it(run_mistral_inference)
timed_run_comprehendit_inference = evaluation.time_it(run_comprehendit_inference)

def main(args):

    ### Configure Data Paths ###
    default_data_dir = os.path.join('.', 'data', 'MedMCQA')
    data_dir = args.data_dir if args.data_dir else default_data_dir
    train_file = os.path.join(data_dir, "train.json")
    dev_file = os.path.join(data_dir, "dev.json") # Used for FLAN validation
    #test_file = os.path.join(data_dir, "test.json")
    test_file = os.path.join(data_dir, "dev.json")

    os.makedirs(args.output_dir, exist_ok=True)
    results = {} 

    ### FLAN-T5 Fine-tuning ###
    if args.train_flan:
        if not os.path.exists(train_file) or not os.path.exists(dev_file):
             logging.error(f"FLAN training needs train.json and dev.json in '{data_dir}'")
        else:
            logging.info(">>> Starting FLAN-T5 Fine-tuning <<<")
            flan_model_out_dir = os.path.join(args.output_dir, "flan_t5_finetuned")
            os.makedirs(flan_model_out_dir, exist_ok=True)
            train_flan_t5( # This function handles its own data loading
                train_file=train_file,
                eval_file=dev_file,
                model_output_dir=flan_model_out_dir,
                num_epochs=args.flan_epochs,
                batch_size=args.flan_batch_size,
                learning_rate=args.flan_lr,
                subject_match=args.subjects
            )
            logging.info(f"FLAN-T5 fine-tuning finished. Model saved to {flan_model_out_dir}")

            if args.eval_flan and not args.flan_model_path:
                 logging.info(f"Setting FLAN eval path to recently trained: {flan_model_out_dir}/final_model")
                 args.flan_model_path = os.path.join(flan_model_out_dir, "final_model")

    ### FLAN-T5 Evaluation ###
    if args.eval_flan:
        # Determine the correct model path to evaluate
        model_path_to_eval = args.flan_model_path
        if not model_path_to_eval:
             trained_path = os.path.join(args.output_dir, "flan_t5_finetuned", "final_model")
             if os.path.exists(trained_path): model_path_to_eval = trained_path

        if model_path_to_eval and os.path.exists(model_path_to_eval) and os.path.exists(test_file):
            logging.info(f">>> Evaluating Fine-tuned FLAN-T5 ({model_path_to_eval}) <<<")
            flan_preds, flan_gt, flan_pred_text, flan_gt_text = timed_run_flan_inference(
                model_path=model_path_to_eval, test_file=test_file,
                batch_size=args.flan_batch_size, subject_match=args.subjects, device=DEVICE
            )
            if flan_preds is not None:
                em = evaluation.calculate_exact_match(flan_preds, flan_gt)
                mrr = evaluation.calculate_mrr(flan_preds, flan_gt)
                bert_f1 = evaluation.calculate_bert_score(flan_pred_text, flan_gt_text, device=DEVICE)
                results["FLAN-T5 Fine-tuned"] = {"EM": em, "MRR": mrr, "BERTScore F1": bert_f1}
                logging.info(f"FLAN-T5 Results: EM={em:.2f}%, MRR={mrr:.4f}, BERTScore F1={bert_f1:.2f}%")
        else:
             logging.error(f"Skipping FLAN evaluation. Check model path ({model_path_to_eval}) and test file ({test_file}).")


    ### Mistral Zero-Shot Evaluation ###
    if args.eval_mistral:
        if os.path.exists(test_file):
            logging.info(f">>> Evaluating Mistral Zero-Shot ({args.mistral_model_id}) <<<")
            try:
                mistral_model, mistral_tokenizer = load_mistral_model(args.mistral_model_id, use_4bit=not args.mistral_no_4bit)
                mistral_preds, mistral_gt, mistral_pred_text, mistral_gt_text = timed_run_mistral_inference(
                    model=mistral_model, tokenizer=mistral_tokenizer, test_file=test_file,
                    batch_size=args.mistral_batch_size, subject_match=args.subjects, device=DEVICE
                )
                if mistral_preds is not None:
                    em = evaluation.calculate_exact_match(mistral_preds, mistral_gt)
                    mrr = evaluation.calculate_mrr(mistral_preds, mistral_gt)
                    bert_f1 = evaluation.calculate_bert_score(mistral_pred_text, mistral_gt_text, device=DEVICE)
                    results["Mistral Zero-Shot"] = {"EM": em, "MRR": mrr, "BERTScore F1": bert_f1}
                    logging.info(f"Mistral Results: EM={em:.2f}%, MRR={mrr:.4f}, BERTScore F1={bert_f1:.2f}%")
                # Clean up GPU memory
                del mistral_model, mistral_tokenizer
                if DEVICE == 'cuda': torch.cuda.empty_cache()
            except Exception as e:
                 logging.error(f"Mistral evaluation failed: {e}", exc_info=True)
        else:
             logging.error(f"Skipping Mistral evaluation. test.json not found in '{data_dir}'.")


    ### ComprehendIt Zero-Shot Evaluation ###
    if args.eval_comprehendit:
         if os.path.exists(test_file):
            logging.info(">>> Evaluating ComprehendIt Zero-Shot <<<")
            try:
                comprehendit_pipeline = load_comprehendit_pipeline(device_num=GPU_INDEX)
                comp_preds, comp_gt, comp_pred_text, comp_gt_text = timed_run_comprehendit_inference(
                    classifier=comprehendit_pipeline, test_file=test_file,
                    batch_size=args.comprehendit_batch_size, subject_match=args.subjects
                )
                if comp_preds is not None:
                    em = evaluation.calculate_exact_match(comp_preds, comp_gt)
                    mrr = evaluation.calculate_mrr(comp_preds, comp_gt)
                    bert_f1 = evaluation.calculate_bert_score(comp_pred_text, comp_gt_text, device=comprehendit_pipeline.device)
                    results["ComprehendIt Zero-Shot"] = {"EM": em, "MRR": mrr, "BERTScore F1": bert_f1}
                    logging.info(f"ComprehendIt Results: EM={em:.2f}%, MRR={mrr:.4f}, BERTScore F1={bert_f1:.2f}%")
                # Clean up memory
                del comprehendit_pipeline
                if DEVICE == 'cuda': torch.cuda.empty_cache()
            except Exception as e:
                logging.error(f"ComprehendIt evaluation failed: {e}", exc_info=True)
         else:
             logging.error(f"Skipping ComprehendIt evaluation. test.json not found in '{data_dir}'.")

    ### Final Results ###
    logging.info("\n--- Evaluation Summary ---")
    if results:
        results_df = pd.DataFrame.from_dict(results, orient='index')
        print(results_df.round(4).to_markdown()) # Round for display
        results_file = os.path.join(args.output_dir, "evaluation_results.csv")
        try:
            results_df.to_csv(results_file)
            logging.info(f"Results saved to {results_file}")
        except Exception as e:
            logging.error(f"Failed to save results to CSV: {e}")
    else:
        logging.info("No evaluation results generated.")


if __name__ == "__main__":
    ### Argument Parser Setup ###

    parser = argparse.ArgumentParser(description="Run MCQA Model Training and Evaluation")

    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='./mcqa_output')
    parser.add_argument('--subjects', nargs='+', default=None)
    parser.add_argument('--train-flan', action='store_true')
    parser.add_argument('--eval-flan', action='store_true')
    parser.add_argument('--flan-model-path', type=str, default=None)
    parser.add_argument('--flan-epochs', type=int, default=3)
    parser.add_argument('--flan-batch-size', type=int, default=8)
    parser.add_argument('--flan-lr', type=float, default=2e-5)
    parser.add_argument('--eval-mistral', action='store_true')
    parser.add_argument('--mistral-model-id', type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument('--mistral-batch-size', type=int, default=2)
    parser.add_argument('--mistral-no-4bit', action='store_true')
    parser.add_argument('--eval-comprehendit', action='store_true')
    parser.add_argument('--comprehendit-batch-size', type=int, default=16)

    args = parser.parse_args()
    main(args)