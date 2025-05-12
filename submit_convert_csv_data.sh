#!/bin/bash
#SBATCH -p compute                          # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 128                      # Specify number of nodes and processors per task
#SBATCH --ntasks-per-node=1		        # Specify number of tasks per node
#SBATCH -t 2:00:00                    # Specify maximum time limit (hour: minute: second)
#SBATCH -A <project_name>               # Specify project name
#SBATCH -J llm_finetuning               # Specify job name

ml reset
ml Mamba
conda deactivate
conda activate ./env

# Set the input file (test.csv)
INPUT_FILE="example/test.csv"

# Set the output directory for the datasets
OUTPUT_DIR="data/processed"

# Ensure the output directory exists
mkdir -p $OUTPUT_DIR

# Run the conversion script for all modes (sft, sft_lora, grpo)
echo "Converting CSV data to TRL datasets..."
python scripts/csv_to_trl_dataset.py \
    --input_file $INPUT_FILE \
    --output_dir $OUTPUT_DIR \
    --val_split 0.1

echo "Conversion complete. Datasets saved to $OUTPUT_DIR"