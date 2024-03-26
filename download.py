from huggingface_hub import snapshot_download
  
model_dir = snapshot_download('GritLM/GritLM-7B',
                              cache_dir='/remote-home/share/LLM_model/',
                              revision='main')
