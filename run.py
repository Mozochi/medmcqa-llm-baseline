from utilities import *
from utilities.data_utils import _df_to_csv
from models import vector_embedding
from lmm_baseline import main, gui
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

#df = json_to_dataframe(r'C:\.repos\python\pytorch\NLP-assign\medmcqa-llm-baseline\data\MedMCQA\dev.json')




gui.interface.launch()