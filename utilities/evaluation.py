import numpy as np
from bert_score import score as bert_score_calc
import time
import torch
import logging
from functools import wraps

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def time_it(func):
    # Decorator to measure execution time of a function
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        duration = end_time - start_time
        logging.info(f"Function '{func.__name__}' executed in {duration:.4f} seconds.")

        return result
    return wrapper

def calculate_exact_match(predictions, ground_truth):
    # Calculates the exact match score 
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth lists must have the same length")
    
    if not predictions:
        return 0.0
    
    correct_count = sum(p == gt for p, gt in zip (predictions, ground_truth)) #p == prediction gt == ground truth
    return (correct_count / len(predictions)) * 100

def calculate_bert_score(predictions, ground_truth, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Calculates the BERTScore F1.
    if not predictions or not ground_truth:
        return 0.0
    
    logging.warning("BERTScore calculation currently assumes predictions/ground_truth are text.")

    try:
        P, R, F1 = bert_score_calc(predictions, ground_truth, lang="en", model_type='bert-base-uncased', verbose=True, device=device)
        return F1.mean().item() * 100 # Average F1 score * 100
    except Exception as e:
        logging.error(f"Error calculating BERTScore: {e}. Inputs should be list of strings.")
        logging.error(f"Predictions sample: {predictions[:5]}")
        logging.error(f"Ground truth sample: {ground_truth[:5]}")
        return 0.0
    
def calculate_mrr(predictions, ground_truth):
    # CaLculates Mean Reciprocal Rank
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth lists must have the same length")
    if not predictions:
        return 0.0
    
    reciprocal_rank = []
    for p, gt in zip(predictions, ground_truth): #p == prediction gt == ground truth
        if p == gt:
            reciprocal_rank.append(1.0)
        else:
            reciprocal_rank.append(0.0)
    
    return np.mean(reciprocal_rank) if reciprocal_rank else 0.0

