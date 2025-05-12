## Installation

Repository preparation

```bash
git clone https://github.com/new5558/SuperAI_LLM_FineTune_2025
cd ./SuperAI_LLM_FineTune_2025
```

### Install using Conda

```bash
ml reset
ml Mamba
conda deactivate
conda create -p ./env python=3.10.0 -y
conda activate ./env
conda install -c conda-forge mysqlclient pkg-config -y
pip install -e .
```

## Data Preparation

### Converting CSV Data for Training

The repository includes a script to convert CSV files to the Huggingface dataset format required for TRL training.

1. Place your CSV file in a suitable location (default example is in `example/test.csv`)
2. Run the conversion script:

```bash
# Run the conversion
sbatch ./submit_convert_csv_data.sh
```

The script will convert the CSV data into formatted datasets for all training methods (SFT, SFT-LoRA, and GRPO). The resulting datasets will be saved in the `data/processed` directory.

### CSV Format Requirements

Your CSV file should contain at minimum these columns:

- `Class Index`: The label/class for the text (used as the answer)
- `Title`: The title of the text
- `Description`: (Optional for SFT/SFT-LoRA, used in GRPO) The content to classify

## Training Configuration

Before submitting training jobs, you need to modify the `smultinode_*.sh` scripts to specify the correct paths:

### Common Path Settings for All Training Methods

Edit the following paths in the corresponding `smultinode_*.sh` files:

1. **Base Model Path**: Change `--model_name_or_path` to point to your base model

   ```bash
   --model_name_or_path /path/to/your/base/model
   ```

2. **Training Data Path**: Change `--train_data_path` to point to your training dataset (HF Dataset)

   ```bash
   --train_data_path /path/to/your/training/data
   ```

3. **Validation Data Path**: Change `--eval_data_path` to point to your validation dataset (HF Dataset)

   ```bash
   --eval_data_path /path/to/your/validation/data
   ```

4. **Output Directory**: Change `--output_dir` to specify where to save the trained model
   ```bash
   --output_dir /path/to/save/model/checkpoints
   ```

### Specific Settings by Training Method

- **For SFT**: Edit `smultinode_trl_sft.sh`
- **For SFT with LoRA**: Edit `smultinode_trl_sft_lora.sh`
- **For GRPO**: Edit `smultinode_trl_grpo.sh` and `smultinode_trl_grpo_vllm.sh`
  - GRPO additionally uses a vLLM server for inference while training
  - The vLLM server host is automatically determined by the submit script
  - **Important**: Make sure the model path in `smultinode_trl_grpo_vllm.sh` matches the base model path (`--model_name_or_path`) in `smultinode_trl_grpo.sh`

## Submit Train Model

### Supervised Fine-Tuning (SFT)

```bash
sbatch submit_multinode_trl_sft.sh
```

### SFT with LoRA

```bash
sbatch submit_multinode_trl_sft_lora.sh
```

### General Reward-Based Policy Optimization (GRPO)

```bash
sbatch submit_multinode_trl_grpo.sh
```

## Model Inference

### Standard Inference (for SFT and LoRA models)

To run inference with models trained using SFT or LoRA:

```bash
sbatch submit_inference.sh
```

### GRPO Inference

For models trained with GRPO, which use vLLM for faster inference:

```bash
sbatch submit_inference_grpo.sh
```

Note:

- Inference scripts can be found in the `scripts/` directory.
- Both inference scripts use the same command-line arguments:
  ```bash
  # In submit_inference.sh and submit_inference_grpo.sh
  MODEL_DIR="/path/to/your/model/checkpoint"
  VAL_PATH="/path/to/your/validation/data"
  ```
- Model paths should point to your trained model checkpoints.
- Validation paths should point to your validation datasets.
