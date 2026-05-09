#!/bin/bash
#SBATCH --account=lsaie-ss26
#SBATCH --time=00:20:00
#SBATCH --job-name=gipfel-hybrid-tp2-opt
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sofiatoik@gmail.com

echo "START TIME: $(date)"

################ Configs ################
WORKDIR=/users/course_00306/Project/large-whale-ai
MEGATRON_LM_DIR=$WORKDIR/Megatron-LM
DATA_PREFIX=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small
DATASET_CACHE_DIR=/users/course_00306/scratch/gipfelsturm/dataset_cache

mkdir -p $DATASET_CACHE_DIR

# Training config - INCREASED micro-batch size for better efficiency
MBS=4  # Increased from 2
GBS=256
SEQ_LEN=4096
TRAINING_STEPS=250

# TP=2
TP_SIZE=2
PP_SIZE=1

# Logging
EXP_DIR=/users/course_00306/scratch/gipfelsturm/experiments
EXP_NAME=train-8b-hybrid-tp${TP_SIZE}-opt-${SLURM_NNODES}n
LOG_DIR=$EXP_DIR/$EXP_NAME
TENSORBOARD_DIR=$LOG_DIR/tensorboard

#########################################

mkdir -p logs $EXP_DIR $LOG_DIR $TENSORBOARD_DIR $DATASET_CACHE_DIR

cd $MEGATRON_LM_DIR

export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NVLS_ENABLE=0

# Let NCCL auto-tune - remove fixed algorithms
unset NCCL_ALGO
unset NCCL_PROTO
# Allow NCCL to use all available channels
export NCCL_MAX_NCHANNELS=32
export NCCL_NSOCKS_PERTHREAD=8
export NCCL_SOCKET_NTHREADS=4

export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
MASTER_ADDR=$(hostname)
MASTER_PORT=25678

echo "Running with TP=${TP_SIZE}, PP=${PP_SIZE}"
echo "Data parallel size = $(($SLURM_NNODES * $SLURM_GPUS_PER_NODE / ${TP_SIZE}))"
echo "Micro-batch size = ${MBS} (increased from 2)"
echo "NCCL auto-tuning enabled (no fixed ALGO/PROTO)"

srun -lu --mpi=pmix --network=disable_rdzv_get \
     --environment=alps3 \
     --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 \
     bash -c "torchrun \
        --nproc-per-node $SLURM_GPUS_PER_NODE \
        --nnodes $SLURM_NNODES \
        --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
        --rdzv_backend c10d \
        --max_restarts 0 \
        --tee 3 \
        $MEGATRON_LM_DIR/pretrain_gpt.py \
        --transformer-impl transformer_engine \
        --use-precision-aware-optimizer \
        --main-grads-dtype bf16 \
        --tensor-model-parallel-size ${TP_SIZE} \
        --pipeline-model-parallel-size ${PP_SIZE} \
        --use-distributed-optimizer \
        --overlap-grad-reduce \
        --overlap-param-gather \
        --num-layers 32 \
        --hidden-size 4096 \
        --ffn-hidden-size 14336 \
        --num-attention-heads 32 \
        --group-query-attention \
        --num-query-groups 8 \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --position-embedding-type rope \
        --normalization RMSNorm \
        --swiglu \
        --untie-embeddings-and-output-weights \
        --micro-batch-size ${MBS} \
        --global-batch-size ${GBS} \
        --train-iters ${TRAINING_STEPS} \
        --log-interval 1 \
        --eval-interval 250 \
        --eval-iters 0 \
        --cross-entropy-loss-fusion \
        --disable-bias-linear \
        --optimizer adam \
        --dataloader-type single \
        --no-check-for-nan-in-loss-and-grad \
        --manual-gc \
        --manual-gc-interval 50 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --lr 3e-4 \
        --lr-decay-style constant \
        --lr-warmup-iters 20 \
        --seed 41 \
        --init-method-std 0.02 \
        --bf16 \
        --log-throughput \
        --log-progress \
        --tensorboard-dir $TENSORBOARD_DIR \
        --log-timers-to-tensorboard \
        --log-memory-to-tensorboard \
        --tokenizer-type GPT2BPETokenizer \
        --vocab-file $WORKDIR/data/gpt2-vocab.json \
        --merge-file $WORKDIR/data/gpt2-merges.txt \
        --data-path ${DATA_PREFIX} \
        --data-cache-path $DATASET_CACHE_DIR \
        --split 99,1,0 \
        --num-workers 1"

echo "END TIME: $(date)"