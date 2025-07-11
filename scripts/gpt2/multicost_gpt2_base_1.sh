﻿GPUS=(1)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

MASTER_ADDR=localhost
MASTER_PORT=66$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model

# BASE_PATH=path_to_project
BASE_PATH=/mnt/nam_x/tue_x/DSKD

CKPT_TYPE="gpt2"
CKPT_NAME="gpt2-base"

# CKPT_PATH="${BASE_PATH}/model_hub/${CKPT_TYPE}/${CKPT_NAME}"
CKPT_PATH="/mnt/nam_x/tue_x/model_hub/${CKPT_TYPE}/${CKPT_NAME}"

TEACHER_MODEL_TYPE="qwen"
TEACHER_MODEL_NAME="Qwen1.5-1.8B"
TEACHER_MODEL_PATH="/mnt/nam_x/tue_x/model_hub/qwen/Qwen1.5-1.8B"

# data
DATA_DIR="${BASE_PATH}/data/dolly/"

# task
TASK="dual_space_kd_with_cma_ot"

BATCH_SIZE=2
LR=0.0004
GRAD_ACC=2
EVAL_BATCH_SIZE=2
EPOCH=20
KD_RATE=2.5
KD_TEMP=3.0

# distiller
PROJECTOR_CONFIG_PATH="${BASE_PATH}/configs/projector_config.json"
PROJECTOR_LR=0.004
# length
MAX_LENGTH=512
# runtime
PRECISION="bf16"
# PRECISION="fp16"
CRITERION="dual_space_kd_with_cma_ot"
KD_OBJ="forward_kl"  # [forward_kl, reverse_kl, js_divergence, skewed_forward_kl, skewed_reverse_kl, adaptive_kl]

CONFIG="${KD_OBJ}-${PRECISION}"
SETTING=criterion=${CRITERION}__${CONFIG}__teacher=${TEACHER_MODEL_NAME}__kd^rate=${KD_RATE}__kd^temp=${KD_TEMP}__epoch=${EPOCH}__bsz=${BATCH_SIZE}x${GRAD_ACC}x${GPUS_PER_NODE}=$((BATCH_SIZE * GRAD_ACC * GPUS_PER_NODE * NNODES))__lr=${LR}__proj^lr=${PROJECTOR_LR}
SAVE_PATH="${BASE_PATH}/outputs/${CKPT_TYPE}/${CKPT_NAME}/${TASK}/${SETTING}"
SAVE_BEST_N_CKPTS=1
# seed
SEED=10

mkdir -p ${SAVE_PATH}

OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-type ${CKPT_TYPE}"
OPTS+=" --model-path ${CKPT_PATH}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --teacher-model-type ${TEACHER_MODEL_TYPE}"
OPTS+=" --teacher-model-path ${TEACHER_MODEL_PATH}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --gradient-checkpointing"

# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num 1000"
# task
OPTS+=" --task ${TASK}"

# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --num-epochs ${EPOCH}"
OPTS+=" --kd-rate ${KD_RATE}"
OPTS+=" --kd-temperature ${KD_TEMP}"
OPTS+=" --kd-objective ${KD_OBJ}"

# distiller
OPTS+=" --projector-lr ${PROJECTOR_LR}"
OPTS+=" --projector-config-path ${PROJECTOR_CONFIG_PATH}"

# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"

# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"

# To load checkpoints, for example:
# OPTS+=" --load /mnt/nam_x/tue_x/DSKD/outputs/gpt2/gpt2-base/dual_space_kd_with_cma_ot_kb3/criterion=dual_space_kd_with_cma_ot_kb3__reverse_kl-bf16__teacher=Qwen1.5-1.8B__kd^rate=0.7__kd^temp=2.5__epoch=10__bsz=4x2x1=8__lr=0.0003__proj^lr=0.004/epoch1_step1429_loss7.7672_rougel26.1746"

OPTS+=" --precision ${PRECISION}"
OPTS+=" --save-interval 1"
OPTS+=" --eval-interval 1"
OPTS+=" --log-interval 20"
OPTS+=" --save-dir ${SAVE_PATH}"
OPTS+=" --keep-best-n-checkpoints ${SAVE_BEST_N_CKPTS}"
OPTS+=" --criterion ${CRITERION}"

# Add
OPTS+=" --hidden-dim-student 768"
OPTS+=" --hidden-dim-teacher 2048"
OPTS+=" --max-student-len 512"
OPTS+=" --max-teacher-len 512"
OPTS+=" --proj_dim 256"
OPTS+=" --top_k_vocab 300"
OPTS+=" --ot_weight_logits 50.0"
OPTS+=" --ot_weight_hidden 10.0"
OPTS+=" --ce_weight 10.0"


# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
if [[ $PRECISION == "bf16" ]]; then
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
elif [[ $PRECISION == "fp16" ]]; then
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
elif [[ $PRECISION == "fp32" ]]; then
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_fp32.json"
fi
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/code/distillation_final.py ${OPTS}"

${CMD}
# ${CMD} \
# >> ${SAVE_PATH}/train.log 2>&1 &
