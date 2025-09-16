export WANDB_DISABLED=true

TORCHRUN=/mnt/data/miniconda3/bin/torchrun
export NCCL_DEBUG=INFO
export NCCL_LAUNCH_MODE=PARALLEL
export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3

# deepspeed --master_port=45345 \
torchrun --nproc_per_node=gpu --nnodes=$WORLD_SIZE  --node_rank=$RANK  --master_addr=$MASTER_ADDR  --master_port=$MASTER_PORT \
    run.py --from_pretrained $PRIMUS_SOURCE_CHECKPOINT_DIR/$PRIMUS_SOURCE_CHECKPOINT_ITERATION_DIRNAME \
    --dataset qtc_finetune --model_type task_prefix --label_type gt --llm qwen2 --alpha 0.5 --eval_bs 16 --max_input_length 640 \
    --max_steps 2500000 --eval_steps 500 --save_steps 1000 --deepspeed ds_baichuan_conf_bert_distributed.json --bf16 --hidden_size 896 \
    --grad_steps 8 --per_device_train_batch_size 16 --output_dir $PRIMUS_SAVE_CHECKPOINT_DIR --lr 5e-5 \
	--generate_cls --lr_scheduler_type "cosine" --warmup_ratio 0.1 --weight_decay 0.1 --gradient_checkpointing --is_streaming --distill \
    
    