PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}"
echo "REPO_HOME: $REPO_HOME"
# on remote
data_paths="/home/zhanghansong/DATA/LISA-Annotation/rec_jsons_processed/refcoco_train.jsonl" 
image_folders="/home/zhanghansong/DATA/COCO_Train_2014_Image/"
model_path="/home/zhanghansong/MODEL_PARAMS/Qwen2.5-VL-3B-Instruct/"
is_reward_customized_from_vlm_module=True
echo "data_paths: $data_paths"
echo "image_folders: $image_folders"

export EXP_NAME="Qwen2.5-VL-3B-Instruct-rec-lora" # TODO: change this to your own experiment name
TASK_TYPE="rec"
cd ${REPO_HOME}/src/open-r1-multimodal

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export NCCL_IB_DISABLE=0 # Turn to 1 if use eno1
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_SOCKET_IFNAME=eno1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=106
#####            To log the NCCL info in detail                              #########################
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_FILE=/home/zhanghansong/CODES/VLM-R1/highspeed_bs4_nccl_debug.txt
####################################################################################################


# create the run directory and log file
mkdir -p ${REPO_HOME}/runs/${EXP_NAME}/log
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"
# MAX_STEPS=1200 # TODO: change this to your own max steps


# export WANDB_DISABLED=true
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
torchrun --nproc_per_node="2" \
    --nnodes="2" \
    --node_rank="0" \
    --master_addr="61.12.226.93" \
    --master_port="12349" \
  src/open_r1/grpo_jsonl.py \
    --use_vllm False \
    --output_dir ${REPO_HOME}/checkpoints/rl/${EXP_NAME} \
    --resume_from_checkpoint True \
    --model_name_or_path $model_path \
    --data_file_paths $data_paths \
    --image_folders $image_folders \
    --is_reward_customized_from_vlm_module $is_reward_customized_from_vlm_module \
    --task_type $TASK_TYPE \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --num_train_epochs 2 \
    --bf16 \
    --run_name ${EXP_NAME} \
    --data_seed 42 \
    --save_steps 100 \
    --num_generations 2 \
    --max_completion_length 2048 \
    --reward_funcs accuracy format \
    --beta 0.04 \
    --report_to wandb \
    --dataset-name this_is_not_used \
    --deepspeed ${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero2.json \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true

echo "Training completed for ${EXP_NAME}"
