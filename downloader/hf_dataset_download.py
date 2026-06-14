import os
from huggingface_hub import snapshot_download
from datasets import load_dataset

# ===================== 配置区（按需修改）=====================
# 示例1：完整Hub数据集 id = "cornell-movie-review-data/rotten_tomatoes"
# 示例2：内置数据集短名 id = "rotten_tomatoes"
DATASET_ID = "cornell-movie-review-data/rotten_tomatoes"
# 本地保存文件夹
LOCAL_SAVE_DIR = "./hf_datasets/" + DATASET_ID.replace("/", "_")
# ==========================================================

# 1. 全局设置HF国内镜像（必须最先执行）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "0"


def download_hf_dataset(dataset_id: str, local_dir: str):
    """
    通用Hugging Face数据集下载函数
    :param dataset_id: Hub数据集ID，如 "xxx/yyy" 或内置短名 "rotten_tomatoes"
    :param local_dir: 本地存放目录
    :return: datasets.Dataset 对象
    """
    # 判断是否为带 / 的完整仓库ID（社区数据集，需要cli下载仓库文件）
    if "/" in dataset_id:
        print(f"检测为Hub远程数据集：{dataset_id}")
        print(f"目标本地目录：{local_dir}")

        # 目录不存在则完整下载
        if not os.path.exists(local_dir) or len(os.listdir(local_dir)) == 0:
            print("本地无数据，开始从镜像下载...")
            snapshot_download(
                repo_id=dataset_id,
                repo_type="dataset",
                local_dir=local_dir,
                force_download=False,
                max_workers=8   # 多线程加速下载
            )
            print("✅ 数据集下载完成！")
        else:
            print("✅ 本地已存在数据集，跳过下载")

        # 加载本地文件夹数据集
        ds = load_dataset(local_dir)

    else:
        # 无 / = datasets内置数据集，无需下载仓库，直接加载
        print(f"检测为内置数据集 {dataset_id}，直接在线加载缓存")
        ds = load_dataset(dataset_id)

    return ds


if __name__ == "__main__":
    # 执行下载+加载
    dataset = download_hf_dataset(DATASET_ID, LOCAL_SAVE_DIR)

    # 打印数据集基础信息
    print("\n===== 数据集信息 =====")
    print(f"全部划分：{list(dataset.keys())}")
    print(f"训练集样本量：{len(dataset['train'])}")
    print(f"单条样本示例：{dataset['train'][0]}")