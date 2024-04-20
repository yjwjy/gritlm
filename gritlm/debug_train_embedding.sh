# torchrun --nproc_per_node 1 \
export WANDB_MODE=disabled
# export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node 1 \
-m debugpy --listen 5678 --wait-for-client -m training.run \
--output_dir training/msmarco_train \
--model_name_or_path /home/wjy/Mistral-7B-Instruct-v0.1 \
--train_data training/msmarco_train.jsonl \
--learning_rate 1e-5 \
--num_train_epochs 5 \
--per_device_train_batch_size 4 \
--dataloader_drop_last True \
--normalized True \
--temperature 0.02 \
--query_max_len 32 \
--passage_max_len 144 \
--train_group_size 2 \
--mode embedding \
--attn cccc \
--lora \
--save_step 10 \
--logging_steps 10 \

