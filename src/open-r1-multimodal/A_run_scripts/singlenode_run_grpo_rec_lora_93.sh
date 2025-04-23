cd src/open-r1-multimodal
# export NCCL_IB_DISABLE=0
# export NCCL_SOCKET_IFNAME=eno1 
export DEBUG_MODE="false"


export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_SOCKET_IFNAME=eno1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=106

# export NCCL_DEBUG_FILE=/home/zhanghansong/CODES/VLM-R1/nccl_debug.txt
# export NCCL_DEBUG_SUBSYS=ALL
export NCCL_DEBUG=WARN

# export CUDA_VISIBLE_DEVICES=4,5,6,7

RUN_NAME="Qwen2.5-VL-3B-GRPO-REC-lora"
export LOG_PATH="./log_$RUN_NAME.txt"
# export WANDB_INSECURE='true'


# nsys profile \
#   --trace=cuda,nvtx,cublas,osrt \
#   --output=report \
#   --force-overwrite=true \
#   --trace-fork-before-exec=true \
#   --stop-on-exit=false \
torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="61.12.226.93" \
    --master_port="12346" \
    src/open_r1/grpo_rec.py \
    --deepspeed local_scripts/zero2.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path /home/zhanghansong/MODEL_PARAMS/Qwen2.5-VL-3B-Instruct/ \
    --dataset_name data_config/rec.yaml \
    --image_root /home/zhanghansong/DATA/COCO_Train_2014_Image/ \
    --max_prompt_length 1024 \
    --num_generations 2 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true


