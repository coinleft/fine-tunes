#!/usr/bin/env python3
"""
HF Snapshot Download Script
从 Hugging Face Hub 下载完整模型/数据集仓库。

依赖: pip install huggingface_hub tqdm
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Optional, List

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError
except ImportError:
    print("请先安装依赖: pip install huggingface_hub")
    sys.exit(1)

try:
    from tqdm.contrib.logging import logging_redirect_tqdm
except ImportError:
    logging_redirect_tqdm = None


def setup_logging(verbose: bool = False):
    """配置日志输出"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("hf_download")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="下载 Hugging Face Hub 上的模型/数据集仓库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载整个仓库到默认缓存目录
  python hf_download.py --repo_id bert-base-chinese

  # 下载到指定目录
  python hf_download.py --repo_id Qwen/Qwen2-7B --local_dir ./qwen2-7b

  # 只下载特定文件（如 safetensors 和配置文件）
  python hf_download.py --repo_id meta-llama/Llama-2-7b --pattern "*.safetensors" "*.json" "*.model"

  # 排除某些文件
  python hf_download.py --repo_id bigscience/bloom --ignore "*.msgpack" "*.h5"

  # 下载指定分支/版本
  python hf_download.py --repo_id THUDM/chatglm3-6b --revision main
        """
    )

    parser.add_argument(
        "--repo_id",
        required=True,
        help="Hugging Face 仓库 ID，例如: bert-base-chinese 或 org/model",
    )
    parser.add_argument(
        "--repo_type",
        choices=["model", "dataset", "space"],
        default="model",
        help="仓库类型 (默认: model)",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default=None,
        help="下载到本地指定目录（如果不指定，使用默认缓存目录）",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Hugging Face 缓存目录",
    )
    parser.add_argument(
        "--pattern",
        nargs="+",
        default=None,
        help="允许下载的文件模式，例如: '*.safetensors' '*.json'",
    )
    parser.add_argument(
        "--ignore",
        nargs="+",
        default=None,
        help="忽略的文件模式，例如: '*.msgpack' '*.bin'",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face access token (或设置 HF_TOKEN 环境变量)",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="仅使用本地缓存，不连接网络",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="并行下载线程数 (默认: 8)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细日志",
    )

    return parser.parse_args()


def download_snapshot(
    repo_id: str,
    repo_type: str = "model",
    local_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
    token: Optional[str] = None,
    local_files_only: bool = False,
    max_workers: int = 8,
    logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    """
    执行 snapshot_download 下载
    """
    logger = logger or logging.getLogger("hf_download")

    # 优先使用环境变量中的 token
    token = token or os.environ.get(
        "HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    if token:
        logger.info("使用提供的 access token 进行认证")
    else:
        logger.info("未提供 token，尝试以未认证方式下载（仅限公开仓库）")

    # 构建本地目录
    if local_dir:
        local_path = Path(local_dir).expanduser().resolve()
        local_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"目标目录: {local_path}")
    else:
        local_path = None
        logger.info("使用默认缓存目录")

    logger.info(f"开始下载: {repo_id} (type={repo_type})")

    if allow_patterns:
        logger.info(f"允许模式: {allow_patterns}")
    if ignore_patterns:
        logger.info(f"忽略模式: {ignore_patterns}")

    try:
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=str(local_path) if local_path else None,
            cache_dir=cache_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            token=token,
            local_files_only=local_files_only,
            max_workers=max_workers,
        )

        logger.info(f"下载完成: {downloaded_path}")
        return downloaded_path

    except RepositoryNotFoundError:
        logger.error(
            f"仓库未找到: {repo_id}。请检查 ID 是否正确，或需要提供 access token 访问私有仓库。")
        return None
    except Exception as e:
        logger.error(f"下载失败: {type(e).__name__}: {e}")
        return None


def main():
    args = parse_args()
    logger = setup_logging(args.verbose)

    # 如果支持 tqdm，重定向日志避免冲突
    log_ctx = logging_redirect_tqdm if logging_redirect_tqdm else lambda: __import__(
        'contextlib').nullcontext()

    with log_ctx():
        result = download_snapshot(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            local_dir=args.local_dir,
            cache_dir=args.cache_dir,
            allow_patterns=args.pattern,
            ignore_patterns=args.ignore,
            token=args.token,
            local_files_only=args.local_files_only,
            max_workers=args.max_workers,
            logger=logger,
        )

    if result:
        print(f"\n✅ 下载成功: {result}")
        sys.exit(0)
    else:
        print("\n❌ 下载失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
