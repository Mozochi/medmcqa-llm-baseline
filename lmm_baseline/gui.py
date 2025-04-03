import gradio as gr
from models import flan_tokenize, EmbeddingGenerator
from utilities import data_utils
import pandas
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datasets import Dataset

embedder = EmbeddingGenerator()

dev_dir = r'C:\.repos\python\pytorch\NLP-assign\medmcqa-llm-baseline\data\MedMCQA\dev.json'
training_dir = r'C:\.repos\python\pytorch\NLP-assign\medmcqa-llm-baseline\data\MedMCQA\train.json'


df = data_utils.json_to_dataframe(dev_dir)

# Getting all subject names and adding them to a list
subjects = []
for subject in df['subject_name']:
    if subject not in subjects:
        subjects.append(subject)

    

def baseline(selected_subjects, query):
    
    return 0

def FLAN_T5(selected_subjects):
    df= data_utils.load_and_transform_json(dev_dir, subject_match=selected_subjects)
    dataset = Dataset.from_pandas(df)
    
    tokenized_data = dataset.map(flan_tokenize.preprocess, batched=True)

    train_test_split = tokenized_data.train_test_split(test_size=0.25)
    train_data = train_test_split["train"]
    test_data = train_test_split["test"]

    training_args = TrainingArguments(
    output_dir=r"C:\.repos\python\pytorch\NLP-assign\medmcqa-llm-baseline\data",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

def comprehend_it(selected_subjects, query):
    classifier = pipeline("zero-shot-classification",
                      model="knowledgator/comprehend_it-base")
    possible_labels = ["True", "False"]

    scores = classifier(query, possible_labels)
    
    return scores

baseline_model = gr.Interface(fn=baseline, 
                              inputs=[gr.Dropdown(subjects, multiselect=True, label='Subject'), gr.Textbox(label="Input Text")], 
                              outputs=['text'])

FLAN_T5_model = gr.Interface(fn=baseline, 
                             inputs=[gr.Dropdown(subjects, multiselect=True, label='Subject')], 
                             outputs=['text'])

comprehend_it_model = gr.Interface(fn=comprehend_it, 
                                   inputs=[gr.Dropdown(subjects, multiselect=True, label='Subject'), gr.Textbox(label="Input Text")], 
                                   outputs=['text'])
    
interface = gr.TabbedInterface([baseline_model, FLAN_T5_model, comprehend_it_model], ["Baseline Model", "FLAN T5 Model", "Comprehend it Model"])        

#interface = gr.Interface(fn=main, inputs=[gr.Dropdown(subjects, multiselect=True, label='Subject'), gr.Checkbox(label='embed?')], outputs=['text'])
