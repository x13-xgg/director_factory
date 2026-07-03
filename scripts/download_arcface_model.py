"""下载 ArcFace buffalo_l 模型到 D: 盘项目目录。

用法:
    python scripts/download_arcface_model.py

模型约 330MB，下载后解压到 assets/models/insightface/models/buffalo_l/
下载成功后 EmbedExtractor / FaceConsistencyChecker / CharacterConsistencyChecker
将自动使用真实 ArcFace 嵌入。
"""
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

MODEL_ROOT = Path(__file__).parent.parent / "assets" / "models"
MODEL_DIR = MODEL_ROOT / "insightface" / "models" / "buffalo_l"
ZIP_PATH = MODEL_ROOT / "buffalo_l.zip"

# 多个镜像源 (按优先级排列)
MIRRORS = [
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    "https://ghproxy.com/https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    "https://hub.fastgit.xyz/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
]


def download_with_progress(url: str, dest: Path) -> bool:
    print(f"尝试: {url}")
    try:
        def report(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                print(f"\r  下载中: {pct}% ({downloaded / 1024 / 1024:.0f}/{total_size / 1024 / 1024:.0f} MB)", end="")

        urllib.request.urlretrieve(url, str(dest), reporthook=report)
        print()
        return True
    except Exception as e:
        print(f"\n  失败: {e}")
        return False


def main():
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    # 检查是否已存在
    required_files = ["w600k_r50.onnx", "det_10g.onnx", "2d106det.onnx"]
    if MODEL_DIR.exists() and all((MODEL_DIR / f).exists() for f in required_files):
        print(f"模型已存在: {MODEL_DIR}")
        print("文件:", ", ".join(f for f in required_files if (MODEL_DIR / f).exists()))
        return

    # 尝试下载
    success = False
    for url in MIRRORS:
        if download_with_progress(url, ZIP_PATH):
            success = True
            break

    if not success:
        print("\n所有镜像均失败。请手动下载 buffalo_l.zip 放到以下路径:")
        print(f"  {ZIP_PATH}")
        print("下载地址: https://github.com/deepinsight/insightface/releases/tag/v0.7")
        sys.exit(1)

    # 解压
    print(f"解压到 {MODEL_DIR} ...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(MODEL_DIR)

    # 清理
    ZIP_PATH.unlink()

    # 验证
    missing = [f for f in required_files if not (MODEL_DIR / f).exists()]
    if missing:
        print(f"警告: 缺少文件: {missing}")
    else:
        print(f"安装成功! {len(required_files)} 个模型文件就绪")
        # 设置环境变量提示
        print(f"\n确保环境变量设置: INSIGHTFACE_HOME={MODEL_ROOT}")
        os.environ["INSIGHTFACE_HOME"] = str(MODEL_ROOT)
        print("重新导入 insightface 后将自动使用本地模型。")


if __name__ == "__main__":
    main()
