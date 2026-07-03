"""共享人脸分析模块 — ArcFace 嵌入提取 + 余弦相似度, 带优雅降级"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np

log = logging.getLogger("FaceUtils")

EMBEDDING_DIM = 512

# 模型目录: 优先 D 盘项目内, 其次 insightface 默认路径
_MODEL_ROOT = os.environ.get(
    "INSIGHTFACE_HOME",
    str(Path(__file__).parent.parent.parent / "assets" / "models"),
)

_face_app = None
_face_available: bool | None = None  # None = 未检测, True/False = 已知


def _get_face_app():
    global _face_app, _face_available
    if _face_available is False:
        return None
    if _face_app is not None:
        return _face_app

    try:
        from insightface.app import FaceAnalysis
        os.environ.setdefault("INSIGHTFACE_HOME", _MODEL_ROOT)
        app = FaceAnalysis(
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        _face_app = app
        _face_available = True
        log.info("ArcFace 模型加载成功")
        return _face_app
    except Exception as e:
        log.warning(f"ArcFace 模型加载失败, 回退确定性模拟: {e}")
        _face_available = False
        return None


def extract_embedding(image_path: str) -> np.ndarray | None:
    """从图像提取 512 维人脸嵌入向量。失败返回 None。"""
    app = _get_face_app()
    if app is None:
        return None

    try:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            return None
        faces = app.get(img)
        if not faces:
            return None
        # 取最大人脸
        best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = best.normed_embedding
        return emb.astype(np.float32) if emb is not None else None
    except Exception as e:
        log.warning(f"提取嵌入失败: {image_path} — {e}")
        return None


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """计算两个向量的余弦相似度 [0, 1]"""
    a_np = np.asarray(a, dtype=np.float32)
    b_np = np.asarray(b, dtype=np.float32)
    norm_a = np.linalg.norm(a_np)
    norm_b = np.linalg.norm(b_np)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    sim = float(np.dot(a_np, b_np) / (norm_a * norm_b))
    # 将 [-1, 1] 映射到 [0, 1]
    return (sim + 1.0) / 2.0


def deterministic_embedding(seed: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """生成基于 SHA256 的确定性模拟嵌入向量 (回退用)。"""
    digest = hashlib.sha256(seed.encode()).digest()
    vec = np.zeros(dim, dtype=np.float32)
    for i in range(dim):
        byte_idx = i % len(digest)
        vec[i] = (digest[byte_idx] / 255.0) - 0.5
    # 归一化
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def arcface_available() -> bool:
    """检查 ArcFace 是否可用。"""
    return _get_face_app() is not None
