#!/bin/bash
#SBATCH -p gpu                          # Specify partition [Compute/Memory/GPU]
#SBATCH -N 5 -c 64                      # Specify number of nodes and processors per task
#SBATCH --ntasks-per-node=1		        # Specify number of tasks per node
#SBATCH --gpus-per-node=4		        # Specify total number of GPUs
#SBATCH -t 20:00:00                    # Specify maximum time limit (hour: minute: second)
#SBATCH -A <project_name>               # Specify project name
#SBATCH -J llm_finetuning               # Specify job name

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn 

START=`date`
starttime=$(date +%s)

export WANDB_MODE="offline"

# sent to sub script
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo go $COUNT_NODE
echo $HOSTNAMES

NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
export VLLM_NODE="${NODELIST[4]}"  # Node 4 for vLLM
TRAIN_NODES=$(IFS=,; echo "${NODELIST[*]:0:4}")
echo vllm $VLLM_NODE others $TRAIN_NODES

# srun --nodes=4 --gpus-per-node=4 --nodelist="${TRAIN_NODES}" sh smultinode_trl_grpo.sh &

# # # Run vLLM server on the 5th node (Group 2)
# srun --nodes=1 --gpus-per-node=4 --ntasks-per-node=1 --nodelist="${NODELIST[4]}" sh smultinode_trl_grpo_vllm.sh &
# # srun --nodes=1 --gpus-per-node=4 --ntasks-per-node=1 --nodelist="${NODELIST[4]}" sh smultinode_trl_grpo_vllm.sh
# # srun  --gpus-per-node=4 --ntasks-per-node=1 --gpus-per-task=4  sh smultinode_trl_grpo_vllm.sh &

# wait

###############################################################################
# 1. start vLLM server (background)
###############################################################################
srun --nodes=1 --gpus-per-node=4 --ntasks-per-node=1 \
     --nodelist="${VLLM_NODE}" \
     sh smultinode_trl_grpo_vllm.sh &
VLLM_STEP_PID=$!      # just in case you want to inspect/kill it manually

###############################################################################
# 2. run the training (foreground –- **no &**)
###############################################################################
srun --nodes=4 --gpus-per-node=4 --ntasks-per-node=1 \
     --nodelist="${TRAIN_NODES}" \
     sh smultinode_trl_grpo.sh
TRAIN_RC=$?

###############################################################################
# 3. batch script ends here; Slurm will kill remaining steps automatically
###############################################################################
exit $TRAIN_RC