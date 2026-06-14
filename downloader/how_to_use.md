# 1. 基础下载（默认缓存到 ~/.cache/huggingface/hub）
python hf_download.py --repo_id bert-base-chinese

# 2. 下载到指定目录
python hf_download.py --repo_id Qwen/Qwen2-7B --local_dir ./models/qwen2-7b

# 3. 只下载 safetensors 和配置文件（过滤大文件）
python hf_download.py --repo_id meta-llama/Llama-2-7b \
  --pattern "*.safetensors" "*.json" "*.model" "*.txt"

# 4. 下载数据集
python hf_download.py --repo_id wikitext --repo_type dataset --local_dir ./data/wikitext

# 5. 使用 token 下载私有/Gated 模型
python hf_download.py --repo_id org/private-model --token hf_xxxxxxxx

# 6. 断网时仅使用本地缓存
python hf_download.py --repo_id bert-base-chinese --local_files_only