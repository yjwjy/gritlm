# export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node 1 \
-m training.run \
--output_dir test_path \
--model_name_or_path /home/wjy/Mistral-7B-Instruct-v0.1 \
--train_data training/msmarco_train.jsonl \
--learning_rate 1e-5 \
--num_train_epochs 1 \
--per_device_train_batch_size 2 \
--dataloader_drop_last True \
--normalized True \
--temperature 0.02 \
--query_max_len 32 \
--passage_max_len 128 \
--train_group_size 2 \
--embedding_mode sparse \
--mode embedding \
--attn cccc \
--logging_step 50 \
--save_step 20000 \
--lora
