# Setup

## **1. Prerequisites:**
Python 3.9+

 **Recommended:** A NVIDIA GPU with CUDA installed for reasonable performance, especially for training and Mistral inference. 
 <br/>
 <br/>

 ## **2. Clone the Repo**
 ```bash
 git clone https://github.com/Mozochi/medmcqa-llm-baseline 
```
<br/>
<br/>

## **3. Create Virtual Environment (Recommended):**
```bash
python -m venv venv
```
### Activate (Linux/macOS)
```bash
source venv/bin/activate
```
### Activate (Windows - Git Bash/WSL)
```bash
source venv/Scripts/activate
```
### Activate (Windows - Cmd/PowerShell)
```bash
.\venv\Scripts\activate
```
<br/>
<br/>

## **4. Install Dependencies**
```bash
pip install -r requirements.txt
```
<br/>
<br/>

## **5. Data**
Download the MedMCQA dataset.
Make sure the dev.json, test.json, and train.json files are placed inside the data/MedMCQA/ directory. 

<br/>
<br/>

## **Usage**
All commands should be run from the root directory of the project. 
<br/>
### **Training FLAN-T5**
This fine-tunes the Google FLAN-T5 small model on the 'train.json' dataset.
bash
Command:
```bash
python evaluate_pipeline.py --train-flan --output-dir ./mcqa_output --flan-epochs 3 --flan-batch-size 8
```
Options:
* --train-flan: Flag to enable training.
* --output-dir ./mcqa_output: Directory where the trained model and logs will be saved.
* --flan-epochs 3: Number of training epochs (adjust as needed).
* --flan-batch-size 8: Training batch size (adjust based on GPU memory).
* --flan-lr 2e-5: Learning rate (optional, defaults available).
* --subjects Subject1 Subject2: (Optinoal) Train only on specific subjects. 

<br/>
<br/>

## Running evaluation
These commands run inference on the 'test.json' dataset and calculate performance metrics. Results are printed and also saved to 'evaluation_results.csv' in the output directory. 

<br/>

### Evaluate the Fine-tuned FLAN-T5:
(Requires the training of the model (saved previously))
Command:
```bash
python evaluate_pipeline.py --eval-flan --flan-model-path ./mcqa_output/flan_t5_finetuned/final_model --output-dir ./mcqa_output
```
Options:
* --eval-flan: Flag to enable FLAN evaluation.
* --flan-model-path ...: Crucial: Path to your saved fine-tuned FLAN model.

<br/>

### Evaluate Mistral (Zero-Shot):
Command (Using default 4-bit quantization):
```bash
python evaluate_pipeline.py --eval-mistral --output-dir ./mcqa_output
```
Command (Without 4-bit quantization - needs more VRAM):
```bash
python evaluate_pipeline.py --eval-mistral --mistral-no-4bit --output-dir ./mcqa_output
```
Options:
* --eval-mistral: Flag to enable Mistral evaluation.
* --mistral-model-id ...: (Optional) Specify a different Mistral model ID if needed.
* --mistral-no-4bit: Disable 4-bit quantization. 

<br/>

### Evaluate ComprehendIt (Zero-Shot):
Command:
```bash
python evaluate_pipeline.py --eval-comprehendit --output-dir ./mcqa_output
```
Options:
* --eval-comprehendit: Flag to enable ComprehendIt evaluation. 

<br/>

### Evaluate Multiple Models at Once:
Command:
```bash
python evaluate_pipeline.py --eval-flan --flan-model-path ./mcqa_output/flan_t5_finetuned/final_model --eval-mistral --eval-comprehendit --output-dir ./mcqa_output
```

<br/>

### Evaluate on Specific Subjects:
Command:
```bash
python evaluate_pipeline.py --eval-mistral --output-dir ./mcqa_output --subjects Cardiology Neurology
```
Options:
* --subjects ...: Filters the `test.json` data.

<br/>
<br/>

## **Running the Gradio Web UI**
This launches an interactive interface in your browser.

<br/>

### Basic Launch:
Command:
```bash
python run.py
```
Notes:
Uses default paths (e.g., `./mcqa_output` for results CSV, `./mcqa_output/flan_t5_finetuned/final_model` for testing FLAN).
Access the local URL printed in the console (usually `http://127.0.0.1:7860`).

<br/>

### Launch with Custom Paths:
Command:
```bash
python run.py --output-dir /path/to/results --flan-model-path /path/to/your/flan_model --data-dir /path/to/data
```
Notes:
Use these flags if your results, trained FLAN model, or MedMCQA data are not in the default locations. 

<br/>
<br/>

## **Results**
Results

Evaluation results are:
1. Printed to the console when running `evaluate_pipeline.py`.
2. Saved to `evaluation_results.csv` within the specified `--output-dir`.
3. Viewable in the "Summary Results" tab of the Gradio UI (after loading).

<br/>

### Sample Results:
Running on a I5-13600k and a RTX 4080:

| Method                   | EM (%) | MRR    | BERTScore F1 (%) | Inference Time (HH:MM:SS) | Training Time (HH:MM:SS) |
| :----------------------- | :----- | :----- | :--------------- | :------------------------ | :------------------------|
| Fine-tuned FLAN-T5 Small | 34.33  | 0.34   | 71.79            |  00:01:20                 | 01:27:05                 |
| Zero-shot Mistral-7B     | 36.82  | 0.37   | 71.43            |  00:37:41                 | N/A                      |
| Zero-shot ComprehendIt   | 27.92  | 0.28   |  68.43           |  00:08:53                 | N/A                      |

<br/>
<br/>


## **Acknowledgements**

This project uses the MedMCQA dataset (https://medmcqa.github.io/) by Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. Please cite their work if you use the dataset.
Built using Hugging Face Transformers (https://github.com/huggingface/transformers), PyTorch (https://pytorch.org/), and Gradio (https://www.gradio.app/).
