from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_from_disk
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from llm_finetune.grpo import SYSTEM_PROMPT_REASONING

import argparse

BATCH_SZ  = 32
MAX_GEN   = 1024

def parse_args():
    parser = argparse.ArgumentParser(description="Script for model path and validation data path.")
    parser.add_argument('--model_dir', type=str,
                        help='Path to the model directory')
    parser.add_argument('--val_path', type=str,
                        help='Path to the validation data')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir,
                                            padding_side="right",
                                            use_fast=False)

    def build_prompt(input_train):
        """Create a single chat prompt exactly like before."""
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT_REASONING},
            {"role": "user",   "content": input_train},
        ]
        return tokenizer.apply_chat_template(prompt,
                                            tokenize=False,
                                            add_generation_prompt=True,
                                            enable_thinking=True)

    llm = LLM(model             = args.model_dir,
            dtype             = "bfloat16",
            gpu_memory_utilization = 0.90,     # optional
            trust_remote_code = True)          # allow custom code if needed

    sampling = SamplingParams(max_tokens=MAX_GEN,
                            temperature=0.0,   # greedy → deterministic answers
                            top_p=1.0)

    ds = load_from_disk(args.val_path)
    true_labels = ds["answer"]

    def extract_xml_answer(text: str) -> str:
        # exactly your earlier logic
        return text.split("<answer>")[-1].split("</answer>")[0].strip()

    def batched_generation(dataset, batch_size=BATCH_SZ):
        all_preds = []
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i:i+batch_size]
            prompts = [build_prompt(input_train) for input_train in batch['input_train']]
            # vLLM generates a list[RequestOutput]
            outputs = llm.generate(prompts, sampling)
            all_preds.extend([extract_xml_answer(out.outputs[0].text) for out in outputs])
        return all_preds

    predicted_labels = batched_generation(ds)

    enc = LabelEncoder().fit(true_labels + predicted_labels)
    y_true, y_pred = enc.transform(true_labels), enc.transform(predicted_labels)
    prec = precision_score(y_true, y_pred, average="macro")
    rec  = recall_score(y_true, y_pred,    average="macro")
    f1   = f1_score(y_true, y_pred,       average="macro")

    print(f"Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")