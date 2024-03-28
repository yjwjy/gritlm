CUDA_VISIBLE_DEVICES=0,1 python evaluation/eval_mteb.py \
--model_name_or_path /home/data0/wjy/e5-mistral-7b-instruct \
--instruction_set e5 \
--instruction_format gritlm \
--task_names MSMARCO \
--batch_size 64 \
--pipeline_parallel \
--attn_implementation sdpa \
--attn cccc \
--pooling_method mean