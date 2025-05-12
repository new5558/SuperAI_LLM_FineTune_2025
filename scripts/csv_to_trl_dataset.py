#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CSV to TRL Dataset Converter

This script reads a CSV file and converts it to a format compatible with TRL training
for supervised fine-tuning (SFT), SFT with LoRA, and General Reward-Based Policy Optimization (GRPO).

Usage:
    python csv_to_trl_dataset.py --input_file path/to/test.csv --output_dir path/to/output --mode [sft|sft_lora|grpo]
"""

import os
import argparse
import pandas as pd
from datasets import Dataset, DatasetDict
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Convert CSV data to TRL dataset format")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the processed dataset")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def read_csv(file_path):
    """Read the CSV file and return a pandas DataFrame."""
    logger.info(f"Reading CSV file from {file_path}")
    df = pd.read_csv(file_path)
    return df


def create_example(row):
    input_train = f"Title: {row['Title']}\nDescription: {row['Description']}\n"
    answer = str(row['Class Index'])
    
    return {
        "input_train": input_train,
        "answer": answer
    }

def prepare_dataset(df, mode, val_split=0.1, seed=42):
    """
    Prepare dataset for the specified training mode.
    
    Args:
        df: pandas DataFrame with the CSV data
        mode: Training mode (sft, sft_lora, or grpo)
        val_split: Ratio for validation split
        seed: Random seed for reproducibility
    
    Returns:
        DatasetDict with train and validation splits
    """
    logger.info(f"Preparing dataset for {mode} mode")
    
    # Handle missing values
    df = df.fillna("")
    
    examples = [create_example(row) for _, row in df.iterrows()]
    
    # Create a Dataset object
    dataset = Dataset.from_dict({
        key: [example[key] for example in examples]
        for key in examples[0].keys()
    })
    
    # Split into train and validation sets
    random.seed(seed)
    dataset = dataset.shuffle(seed=seed)
    
    split_dataset = dataset.train_test_split(test_size=val_split)
    
    # Rename 'test' split to 'validation'
    dataset_dict = DatasetDict({
        'train': split_dataset['train'],
        'validation': split_dataset['test']
    })
    
    return dataset_dict

def save_dataset(dataset_dict, output_dir):
    """Save the dataset to disk."""
    logger.info(f"Saving dataset to {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    dataset_dict['train'].save_to_disk(os.path.join(output_dir, 'train'))
    dataset_dict['validation'].save_to_disk(os.path.join(output_dir, 'val'))
    
    return output_dir

def main():
    args = parse_args()
    
    # Read the CSV file
    df = read_csv(args.input_file)

    # Prepare the dataset
    dataset_dict = prepare_dataset(df, args.val_split, args.seed)
    
    # Save the dataset
    output_path = save_dataset(dataset_dict, args.output_dir)
    logger.info(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    main() 