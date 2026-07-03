"""下载 ChatTTS 模型到本地项目目录。

用法:
    python scripts/download_chattts_model.py

模型约 2GB，下载后放到 assets/models/chattts/
下载成功后设置 TTS_PROVIDER=chattts 即可使用本地 ChatTTS。

多源下载策略:
    1. ModelScope (国内友好) — modelscope.cn
    2. HuggingFace 镜像 (hf-mirror.com)
    3. HuggingFace 官方 (huggingface.co)
"""
import os
import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).parent.parent / "assets" / "models" / "chattts"
MODEL_ROOT.mkdir(parents=True, exist_ok=True)


def download_from_modelscope():
    """从 ModelScope 下载 (国内最快)"""
    try:
        from modelscope import snapshot_download
        print("尝试: ModelScope (modelscope.cn)")
        path = snapshot_download(
            "pzc163/ChatTTS",
            cache_dir=str(MODEL_ROOT),
        )
        print(f"下载完成: {path}")
        return True
    except ImportError:
        print("  modelscope 未安装, pip install modelscope")
        return False
    except Exception as e:
        print(f"  ModelScope 失败: {e}")
        return False


def download_from_hf_mirror():
    """从 HF 镜像下载"""
    try:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        from huggingface_hub import snapshot_download
        print("尝试: HuggingFace 镜像 (hf-mirror.com)")
        path = snapshot_download(
            "2Noise/ChatTTS",
            cache_dir=str(MODEL_ROOT),
            max_workers=2,
        )
        print(f"下载完成: {path}")
        return True
    except Exception as e:
        print(f"  HF 镜像失败: {e}")
        return False


def download_from_hf_official():
    """从 HF 官方下载 (需要科学上网)"""
    try:
        os.environ.pop("HF_ENDPOINT", None)
        from huggingface_hub import snapshot_download
        print("尝试: HuggingFace 官方 (huggingface.co)")
        path = snapshot_download(
            "2Noise/ChatTTS",
            cache_dir=str(MODEL_ROOT),
            max_workers=2,
        )
        print(f"下载完成: {path}")
        return True
    except Exception as e:
        print(f"  HF 官方失败: {e}")
        return False


def main():
    # 检查是否已存在
    required = ["asset/Vocos.safetensors", "asset/DVAE.safetensors",
                 "asset/Embed.safetensors", "asset/Decoder.safetensors",
                 "asset/GPT/config.json", "asset/GPT/model.safetensors"]
    snapshot_dir = None
    for d in MODEL_ROOT.iterdir():
        if d.is_dir() and d.name.startswith("models--"):
            snapshot_dir = d
            break

    if snapshot_dir:
        snapshots = snapshot_dir / "snapshots"
        if snapshots.exists():
            for s in snapshots.iterdir():
                if s.is_dir():
                    exists = all((s / f).exists() for f in required)
                    if exists:
                        print(f"模型已存在: {s}")
                        print("设置 TTS_PROVIDER=chattts 即可使用")
                        return
                    break

    for download_fn in [download_from_modelscope, download_from_hf_mirror, download_from_hf_official]:
        if download_fn():
            print("\nChatTTS 模型下载成功！")
            print("设置环境变量 TTS_PROVIDER=chattts 启用")
            return

    print("\n所有下载源均失败。请手动操作:")
    print("1. pip install modelscope")
    print("2. python -c \"from modelscope import snapshot_download; snapshot_download('pzc163/ChatTTS', cache_dir='assets/models/chattts')\"")
    print(f"或手动下载 ChatTTS 模型文件到: {MODEL_ROOT}")
    sys.exit(1)


if __name__ == "__main__":
    main()
