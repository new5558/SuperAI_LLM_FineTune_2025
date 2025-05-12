from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import torch

from peft import PeftModel, PeftConfig  # Add these imports
from datasets import load_from_disk

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Script for model path and validation data path.")
    parser.add_argument('--model_dir', type=str,
                        help='Path to the model directory')
    parser.add_argument('--val_path', type=str,
                        help='Path to the validation data')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

        # Try to load PEFT config but have fallback for regular models
    try:
        peft_config = PeftConfig.from_pretrained(args.model_dir)
        base_model_name = peft_config.base_model_name_or_path
        is_peft_model = True
        print(f"Found PEFT adapter config, base model: {base_model_name}")
    except Exception as e:
        # If no PEFT config, assume args.model_dir is the base model itself
        base_model_name = args.model_dir
        is_peft_model = False
        print(f"No PEFT config found, treating as regular model: {base_model_name}")
    
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )
    
    # Only attempt to load adapter if we found a valid PEFT config
    if is_peft_model:
        try:
            model = PeftModel.from_pretrained(base_model, args.model_dir)
            print("Successfully loaded PEFT adapter")
        except Exception as e:
            model = base_model
            print(f"Failed to load PEFT adapter: {e}")
    else:
        model = base_model



    # 2. Optimize model for inference
    model.eval()                    # Set model to evaluation mode
    if torch.__version__ >= "2.0.0":
        model = torch.compile(model)  # PyTorch 2.0+ compilation for speed

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        padding_side="right",
        use_fast=False,
    )

    dataset = load_from_disk(args.val_path)



    # Function to extract the actual prediction content from model output
    def clean_output(output):
        # Remove any special tokens or formatting
        # Adjust this based on your model's output format
        if len(output) == 0:
            return '0'
        if '0' in output:
            return '0'
        if '1' in output:
            return '1'
        if '2' in output:
            return '2'
        if '3' in output:
            return '3'
        return '1'

    # To save memory during processing
    @torch.no_grad()
    def process_batch(batch_items, batch_size=8):
        all_predictions = []
        
        for i in tqdm(range(0, len(batch_items), batch_size), total = len(batch_items) // batch_size):
            current_batch = batch_items[i:i+batch_size]
            prompts = [
                [{'role': 'user', 'content': input_train}]
                for input_train in current_batch['input_train']
            ]
            
            texts = [tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            ) for prompt in prompts]
            
            inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=8  # Adjust based on your needs
            )
            
            for j, output in enumerate(outputs):
                input_length = len(inputs.input_ids[j])
                output_ids = output[input_length:].tolist()
                
                try:
                    index = len(output_ids) - output_ids[::-1].index(151668)  # Find </think> token
                except ValueError:
                    index = 0
                    
                # content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
                content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
                print(content, 'content')
                all_predictions.append(clean_output(content))
        
        return all_predictions

    # Process the dataset in batches
    true_labels = dataset['answer']
    predicted_labels = []

    # Get predictions
    predictions = process_batch(dataset)
    print(predictions, 'predictions')

    # Convert text predictions to labels
    # This depends on your specific task type
    # For classification tasks:
    from sklearn.preprocessing import LabelEncoder

    # Assuming your task is classification with text labels
    label_encoder = LabelEncoder()
    label_encoder.fit(true_labels + predictions)
    y_true = label_encoder.transform(true_labels)
    y_pred = label_encoder.transform(predictions)

    # Calculate metrics
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    print(f"{args.model_dir=}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro): {recall_macro:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")