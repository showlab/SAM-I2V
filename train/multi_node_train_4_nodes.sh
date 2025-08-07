#!/usr/bin/env bash

# -----------------------------
# Setup Parameters
# -----------------------------
TXT_NAME=SAM-I2V

NNODES=4
NPROC_PER_NODE=8
VISIBLE_DEVICES=0,1,2,3,4,5,6,7
RDZV_BACKEND=etcd
RDZV_ENDPOINT=192.168.10.13:2399
RDZV_ID=4
CONFIG=configs/i2v-train

CONDA_ENV=sam-i2v

# -----------------------------
# Node0
# -----------------------------
echo "Start training on Node0"
nohup ssh -o StrictHostKeyChecking=no 192.168.10.9 "
  conda activate $CONDA_ENV
  export OMP_NUM_THREADS=4
  export NCCL_TIMEOUT=600
  export CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES}
  cd SAM-I2V/train
  torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=0 \
    --rdzv_backend=$RDZV_BACKEND \
    --rdzv_endpoint=$RDZV_ENDPOINT \
    --rdzv_id=$RDZV_ID \
    train.py \
      -c \"$CONFIG\" \
      --use-cluster 0 \
      --num-gpus \"$NPROC_PER_NODE\" \
      --num-nodes \"$NNODES\"
" >> txt/${TXT_NAME}_node_0.log 2>&1 &


# -----------------------------
# Node1
# -----------------------------
echo "Start training on Node1"
nohup ssh -o StrictHostKeyChecking=no 192.168.10.10 "
  conda activate $CONDA_ENV
  export OMP_NUM_THREADS=4
  export NCCL_TIMEOUT=600
  export CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES}
  cd SAM-I2V/train
  torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=1 \
    --rdzv_backend=$RDZV_BACKEND \
    --rdzv_endpoint=$RDZV_ENDPOINT \
    --rdzv_id=$RDZV_ID \
    train.py \
      -c \"$CONFIG\" \
      --use-cluster 0 \
      --num-gpus \"$NPROC_PER_NODE\" \
      --num-nodes \"$NNODES\"
" >> txt/${TXT_NAME}_node_1.log 2>&1 &


# -----------------------------
# Node2
# -----------------------------
echo "Start training on Node2"
nohup ssh -o StrictHostKeyChecking=no 192.168.10.13 "
  conda activate $CONDA_ENV
  export OMP_NUM_THREADS=4
  export NCCL_TIMEOUT=600
  export CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES}
  cd SAM-I2V/train
  torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=2 \
    --rdzv_backend=$RDZV_BACKEND \
    --rdzv_endpoint=$RDZV_ENDPOINT \
    --rdzv_id=$RDZV_ID \
    train.py \
      -c \"$CONFIG\" \
      --use-cluster 0 \
      --num-gpus \"$NPROC_PER_NODE\" \
      --num-nodes \"$NNODES\"
" >> txt/${TXT_NAME}_node_2.log 2>&1 &


# -----------------------------
# Node3
# -----------------------------
echo "Start training on Node3"
nohup ssh -o StrictHostKeyChecking=no 192.168.10.14 "
  conda activate $CONDA_ENV
  export OMP_NUM_THREADS=4
  export NCCL_TIMEOUT=600
  export CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES}
  cd SAM-I2V/train
  torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=3 \
    --rdzv_backend=$RDZV_BACKEND \
    --rdzv_endpoint=$RDZV_ENDPOINT \
    --rdzv_id=$RDZV_ID \
    train.py \
      -c \"$CONFIG\" \
      --use-cluster 0 \
      --num-gpus \"$NPROC_PER_NODE\" \
      --num-nodes \"$NNODES\"
" >> txt/${TXT_NAME}_node_3.log 2>&1 &


# -----------------------------
# Finish
# -----------------------------
echo "All commands have been launched in background."
