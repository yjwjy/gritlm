CUDA_VISIBLE_DEVICES=4 python evaluation/eval_mteb.py \
--model_name_or_path  /root/Mistral-7B-Instruct-v0.1 \
--instruction_set e5 \
--instruction_format gritlm \
--task_names SciFact \
--batch_size 64 \
--pipeline_parallel \
--attn_implementation sdpa \
--attn cccc \
--pooling_method mean