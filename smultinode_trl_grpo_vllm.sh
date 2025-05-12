#!/usr/bin/env bash
#sleep 30
#fi_info -p efa -t FI_EP_RDM

# HOSTNAMES MASTER_ADDR MASTER_PORT COUNT_NODE are coming from the main script


module restore
module load Mamba
module load cudatoolkit/22.7_11.7
module load gcc/10.3.0

conda deactivate
conda activate ./env

echo myuser=`whoami`
echo COUNT_NODE=$COUNT_NODE
echo LD_LIBRARY_PATH = $LD_LIBRARY_PATH
echo PATH = $PATH
echo which mpicc `which mpicc`
echo HOSTNAMES = $HOSTNAMES
echo hostname = `hostname`
echo MASTER_ADDR= $MASTER_ADDR
echo MASTER_PORT= $MASTER_PORT

H=`hostname`
THEID=`echo -e $HOSTNAMES | python -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`
echo THEID=$THEID
echo SLURM_PROCID=$SLURM_PROCID

export NCCL_TIMEOUT=3600000
export NCCL_BLOCKING_WAIT=0

export NCCL_DEBUG=INFO

# trl vllm-serve --model /project/ai901002-ai25tn/interesting_model/Qwen3-8B --data-parallel-size 4
# smultinode_trl_grpo_vllm.sh
# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID  # Makes each task see only one GPU
echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES
trl vllm-serve --model /project/ai901002-ai25tn/interesting_model/Qwen2.5-7B-Instruct --tensor-parallel-size 4
# trl vllm-serve --model /project/ai901002-ai25tn/interesting_model/Qwen2.5-7B-Instruct --tensor-parallel-size 4