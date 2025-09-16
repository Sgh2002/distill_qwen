export WANDB_DISABLED=true
export NCCL_DEBUG=INFO
export NCCL_LAUNCH_MODE=PARALLEL
export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12345 \
    run_new_no_prompt_qwen3_test.py --from_pretrained /mnt/data/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0___6B/ \
    --dataset qtc_finetune --model_type task_prefix --label_type gt --llm qwen2 --alpha 0.5 --eval_bs 16 --max_input_length 768 \
    --max_steps 250000 --eval_steps 10000 --save_steps 10000 --deepspeed ds_baichuan_conf_bert_distributed.json --bf16 --hidden_size 896 \
    --grad_steps 8 --per_device_train_batch_size 32 --output_dir sunliuying/qtc-llm-gx/tmps_sgh/output/test1 --lr 1e-5 \
	--generate_cls --lr_scheduler_type "cosine" --warmup_ratio 0.05 --weight_decay 0.1 --gradient_checkpointing --is_streaming --distill \
    --use_lora --lora_rank 8 --lora_alpha 32 --lora_dropout_p 0.05