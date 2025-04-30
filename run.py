import argparse
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from lmm_baseline.gui import create_gui, DEFAULT_OUTPUT_DIR, DEFAULT_FLAN_PATH, DEFAULT_MISTRAL_ID
from data_utils import load_and_transform_json





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch MCQA Gradio Interface")

    # Arguments to configure paths and models for the GUI
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory containing evaluation_results.csv')
    parser.add_argument('--data-dir', type=str, default='./data/MedMCQA',
                        help='Directory containing dev.json for random examples')
    parser.add_argument('--flan-model-path', type=str, default=DEFAULT_FLAN_PATH,
                        help='Path to the fine-tuned FLAN-T5 model directory for GUI testing')
    parser.add_argument('--mistral-model-id', type=str, default=DEFAULT_MISTRAL_ID,
                        help='Hugging Face model ID for Mistral GUI testing')
    parser.add_argument('--mistral-no-4bit', action='store_true',
                        help='Disable 4-bit quantization for Mistral in GUI (requires more VRAM)')
    parser.add_argument('--share', action='store_true',
                        help='Create a publicly shareable Gradio link')


    args = parser.parse_args()

    print("Creating Gradio Interface...")
    # Pass the arguments to the create_gui function
    interface = create_gui(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        flan_path=args.flan_model_path,
        mistral_id=args.mistral_model_id,
        mistral_4bit=not args.mistral_no_4bit # Pass boolean flag correctly
    )

    print("Launching Gradio Interface...")
    interface.launch(share=args.share) # Pass share argument to launch

    print("Gradio Interface closed.")