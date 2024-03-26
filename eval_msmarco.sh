CUDA_VISIBLE_DEVICES=0 python evaluation/eval_mteb.py \
# --model_name_or_path /root/.cache/huggingface/hub/models--GritLM--GritLM-7B/snapshots/13f00a0e36500c80ce12870ea513846a066004af \
--model_name_or_path  \
--instruction_set e5 \
--instruction_format gritlm \
--task_names MSMARCO \
--batch_size 64 \
--pipeline_parallel \
--attn_implementation sdpa \
--pooling_method mean