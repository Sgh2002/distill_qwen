export WANDB_DISABLED=true

TORCHRUN=/mnt/data/miniconda3/bin/torchrun
export NCCL_DEBUG=INFO
export NCCL_LAUNCH_MODE=PARALLEL
export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3

deepspeed --master_port=45345 \
    run.py --from_pretrained $PRIMUS_SOURCE_CHECKPOINT_DIR/$PRIMUS_SOURCE_CHECKPOINT_ITERATION_DIRNAME \
    --dataset qtc_finetune --model_type task_prefix --label_type gt --llm qwen2 --alpha 0.5 --eval_bs 16 --max_input_length 8192 \
    --max_steps 2500000 --eval_steps 500 --save_steps 500 --deepspeed ds_baichuan_conf_bert_distributed.json --bf16 --hidden_size 896 \
    --grad_steps 2 --per_device_train_batch_size 4 --output_dir $PRIMUS_SAVE_CHECKPOINT_DIR --lr 1e-5 \
	--generate_cls --lr_scheduler_type "cosine" --warmup_ratio 0.05 --weight_decay 0.1 --gradient_checkpointing --is_streaming \
    --use_lora --lora_rank 8 --lora_alpha 32 --lora_dropout_p 0.05 \