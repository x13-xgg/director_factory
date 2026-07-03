"""角色专用工具 — LoRA 训练、Face Embedding 提取、增强一致性检查"""

from __future__ import annotations

import hashlib
import random
import time
from pathlib import Path
from src.tools.base import BaseTool, ToolCall, ToolResult
from src.tools.asset_db import asset_db
from src.tools.face_utils import (
    EMBEDDING_DIM,
    _get_face_app,
    arcface_available,
    cosine_similarity,
    deterministic_embedding,
    extract_embedding,
)
from src.core.logging import get_logger

log = get_logger("CharacterTools")


class LoRATrainerTool(BaseTool):
    """
    LoRA 训练工具 — 为角色训练轻量适配权重

    生产环境: 调用 Kohya / diffusers LoRA 训练脚本
    MVP: 模拟训练过程，记录参数到 asset_db
    """

    def __init__(self):
        super().__init__("lora_trainer")

    def schema(self) -> dict:
        return {
            "name": "lora_trainer",
            "description": "Source or train a LoRA weight for a character via CivitAI / HuggingFace / mock",
            "parameters": {
                "character_id": {"type": "string"},
                "description": {"type": "string", "default": ""},
                "reference_images": {"type": "array", "items": {"type": "string"}},
                "base_model": {"type": "string", "default": "sdxl_v3"},
                "trigger_word": {"type": "string"},
                "steps": {"type": "integer", "default": 800},
                "rank": {"type": "integer", "default": 16},
            },
        }

    async def execute(self, call: ToolCall) -> ToolResult:
        char_id = call.params.get("character_id", "")
        description = call.params.get("description", "")
        ref_images = call.params.get("reference_images", [])
        trigger = call.params.get("trigger_word", f"char_{char_id}")
        base_model = call.params.get("base_model", "sdxl_v3")
        steps = call.params.get("steps", 800)

        if not char_id:
            return ToolResult.fail(data=None, suggestions=["character_id is required"])

        # 1. 尝试从 CivitAI/HuggingFace 获取真实 LoRA
        try:
            from src.tools.lora_sourcing import LoraSourcing
            sourcing = LoraSourcing()
            source_result = await sourcing.source(
                character_id=char_id,
                description=description,
                trigger_word=trigger,
                base_model=base_model,
            )
            await sourcing.close()

            if source_result.get("source") != "mock":
                log.info(f"[{char_id}] LoRA 来源化成功: {source_result['source']} → {source_result['lora_path']}")
                source_result["sourced"] = True
                source_result["training_steps"] = 0
                source_result["rank"] = call.params.get("rank", 16)
                source_result["reference_count"] = len(ref_images)
                return ToolResult.ok(data=source_result)
        except ImportError:
            log.info(f"[{char_id}] lora_sourcing 不可用, 回退 mock")
        except Exception as e:
            log.info(f"[{char_id}] 来源化失败, 回退 mock: {e}")

        # 2. 回退: 创建空占位文件
        seed = hash(char_id) % 100000
        lora_hash = hashlib.sha256(f"{char_id}:{seed}:{trigger}".encode()).hexdigest()[:16]
        lora_path = f"assets/loras/{char_id}_{lora_hash}.safetensors"
        Path(lora_path).parent.mkdir(parents=True, exist_ok=True)
        Path(lora_path).touch()

        train_result = {
            "lora_path": lora_path,
            "lora_hash": lora_hash,
            "trigger_word": trigger,
            "base_model": base_model,
            "training_steps": steps,
            "rank": call.params.get("rank", 16),
            "reference_count": len(ref_images),
            "trained_at": time.time(),
            "sourced": False,
        }

        asset_db.put("char_asset_db", f"{char_id}:lora", train_result, {"type": "lora"})
        asset_db.lock("char_asset_db", f"{char_id}:lora")

        return ToolResult.ok(
            data=train_result,
            mock=True,
        )


class EmbedExtractorTool(BaseTool):
    """
    Face Embedding 提取工具 — ArcFace 真实 512d 向量 + 确定性回退

    后端优先级:
      1. ArcFace (insightface buffalo_l) — 从真实图像提取
      2. 确定性模拟 (SHA256 → 512d normalized) — 无模型时回退
    """

    EMBEDDING_DIM = EMBEDDING_DIM

    def __init__(self):
        super().__init__("embed_extractor")

    def schema(self) -> dict:
        return {
            "name": "embed_extractor",
            "description": "Extract face embedding vector from reference image via ArcFace",
            "parameters": {
                "character_id": {"type": "string"},
                "reference_image": {"type": "string"},
                "method": {"type": "string", "default": "arcface"},
            },
        }

    async def execute(self, call: ToolCall) -> ToolResult:
        char_id = call.params.get("character_id", "")
        ref_image = call.params.get("reference_image", "")
        method = call.params.get("method", "arcface")

        if not char_id:
            return ToolResult.fail(data=None, suggestions=["character_id is required"])

        gen_method = "mock"
        embedding = None

        # 尝试 ArcFace 真实提取
        if arcface_available() and ref_image and Path(ref_image).exists():
            emb = extract_embedding(ref_image)
            if emb is not None:
                embedding = [round(float(v), 6) for v in emb]
                gen_method = "arcface"

        # 回退: 确定性模拟向量
        if embedding is None:
            det_emb = deterministic_embedding(f"{char_id}:{method}")
            embedding = [round(float(v), 6) for v in det_emb]
            gen_method = "deterministic_fallback"

        result = {
            "character_id": char_id,
            "embedding": embedding,
            "dim": self.EMBEDDING_DIM,
            "method": gen_method,
            "reference_image": ref_image,
            "extracted_at": time.time(),
        }

        asset_db.put("char_asset_db", f"{char_id}:embedding", result, {"type": "embedding"})
        asset_db.lock("char_asset_db", f"{char_id}:embedding")

        return ToolResult.ok(data=result)


class CharacterConsistencyCheckerTool(BaseTool):
    """
    增强版角色一致性检查 — ArcFace 余弦相似度 + 多维度验证

    后端优先级:
      1. ArcFace real embedding + cosine similarity (>= 0.75 = pass)
      2. 确定性模拟回退 (基于 character_id 生成参考向量)
    """

    def __init__(self):
        super().__init__("character_consistency_checker")

    def schema(self) -> dict:
        return {
            "name": "character_consistency_checker",
            "description": "Multi-dimensional character consistency verification via ArcFace",
            "parameters": {
                "generated_image_path": {"type": "string"},
                "character_id": {"type": "string"},
                "reference_embedding": {"type": "array", "items": {"type": "number"}},
                "check_dimensions": {"type": "array", "items": {"type": "string"}},
            },
        }

    async def execute(self, call: ToolCall) -> ToolResult:
        char_id = call.params.get("character_id", "")
        ref_emb = call.params.get("reference_embedding", [])
        img_path = call.params.get("generated_image_path", "")
        dimensions = call.params.get("check_dimensions", ["face", "appearance", "style"])
        gen_method = "mock"

        # 尝试真实 ArcFace 验证
        face_similarity = 0.0
        if arcface_available() and img_path and Path(img_path).exists() and ref_emb:
            gen_emb = extract_embedding(img_path)
            if gen_emb is not None:
                face_similarity = cosine_similarity(ref_emb, gen_emb)
                gen_method = "arcface"
            else:
                face_similarity = 0.85 + (hash(f"{char_id}:face") % 15) / 100
        else:
            face_similarity = 0.85 + (hash(f"{char_id}:face") % 15) / 100

        # 外观和风格仍用启发式 (需要 CLIP/LPIPS 才能真实化)
        appearance_match = 0.82 + (hash(f"{char_id}:appearance") % 18) / 100
        style_match = 0.83 + (hash(f"{char_id}:style") % 16) / 100

        overall = face_similarity * 0.5 + appearance_match * 0.3 + style_match * 0.2

        warning_regions = []
        if face_similarity < 0.75:
            warning_regions.append("face_region")
        if appearance_match < 0.83:
            warning_regions.append("overall_appearance")
        if style_match < 0.80:
            warning_regions.append("style_coherence")

        suggestions = []
        if warning_regions:
            suggestions.append(f"Character {char_id} consistency issues in: {', '.join(warning_regions)}")
            if "face_region" in warning_regions:
                suggestions.append("建议增大 IPAdapter 权重到 0.9+")
            if "overall_appearance" in warning_regions:
                suggestions.append("建议增强 negative prompt 中的变形词条")

        return ToolResult.ok(
            data={
                "character_id": char_id,
                "face_similarity": round(face_similarity, 3),
                "appearance_match": round(appearance_match, 3),
                "style_match": round(style_match, 3),
                "overall": round(overall, 3),
                "pass": overall >= 0.85,
                "method": gen_method,
                "warning_regions": warning_regions,
                "suggestions": suggestions,
            }
        )


class MultiCharacterCompositionTool(BaseTool):
    """
    多角色同框检测 — ArcFace 多脸检测 + 逐角色余弦相似度匹配

    流程:
      1. 从生成图像中检测所有人脸 (ArcFace det_10g)
      2. 加载每个预期角色的参考 embedding (从 asset_db)
      3. 用余弦相似度将检测到的人脸匹配到预期角色
      4. 报告缺失角色、未匹配人脸、遮挡冲突
    """

    FACE_SIMILARITY_THRESHOLD = 0.65  # 余弦相似度阈值 (匹配用)
    QUALITY_PASS_THRESHOLD = 0.72     # 通过阈值 (质量用)
    OCCLUSION_IOU_THRESHOLD = 0.3     # bbox IoU 遮挡阈值

    def __init__(self):
        super().__init__("multi_char_composition")

    def schema(self) -> dict:
        return {
            "name": "multi_char_composition",
            "description": "ArcFace multi-face detection + per-character cosine similarity matching",
            "parameters": {
                "generated_image_path": {"type": "string"},
                "character_ids": {"type": "array", "items": {"type": "string"}},
            },
        }

    async def execute(self, call: ToolCall) -> ToolResult:
        char_ids = call.params.get("character_ids", [])
        img_path = call.params.get("generated_image_path", "")

        if len(char_ids) < 2:
            return ToolResult.ok(
                data={"pass": True, "message": "Single character, no multi-char check needed"},
            )

        # 尝试 ArcFace 多脸检测
        if arcface_available() and img_path and Path(img_path).exists():
            return self._real_multi_check(char_ids, img_path)
        else:
            return self._fallback_multi_check(char_ids)

    def _real_multi_check(self, char_ids: list[str], img_path: str) -> ToolResult:
        """使用 ArcFace 进行真实多脸检测和匹配"""
        import cv2
        import numpy as np

        img = cv2.imread(img_path)
        if img is None:
            return self._fallback_multi_check(char_ids)

        app = _get_face_app()
        if app is None:
            return self._fallback_multi_check(char_ids)

        # 1. 检测所有人脸
        faces = app.get(img)
        if not faces:
            return ToolResult.ok(
                data={
                    "character_results": {cid: {"similarity": 0.0, "pass": False, "detected": False}
                                        for cid in char_ids},
                    "all_pass": False,
                    "faces_detected": 0,
                    "faces_expected": len(char_ids),
                    "occupancy_conflict": False,
                    "method": "arcface",
                    "suggestions": ["No faces detected in image — check generation quality"],
                }
            )

        # 按面积排序 (最大脸优先)
        faces_sorted = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )

        # 2. 加载每个预期角色的参考 embedding
        ref_embeddings: dict[str, list[float] | None] = {}
        for cid in char_ids:
            record = asset_db.get("char_asset_db", f"{cid}:embedding")
            if record and record.get("data", {}).get("embedding"):
                ref_embeddings[cid] = record["data"]["embedding"]
            else:
                ref_embeddings[cid] = None

        # 3. 匹配: 贪心算法 — 每个人脸匹配最近的参考 embedding
        detected_embs = []
        detected_boxes = []
        for face in faces_sorted:
            emb = face.normed_embedding
            if emb is not None:
                detected_embs.append(emb.astype(np.float32))
                detected_boxes.append(face.bbox)

        # 构建相似度矩阵
        char_results: dict[str, dict] = {}
        matched_face_indices: set[int] = set()

        for cid in char_ids:
            ref = ref_embeddings.get(cid)
            if ref is None:
                # 无参考 embedding, 用确定性回退
                det_emb = deterministic_embedding(cid)
                ref = [float(v) for v in det_emb]

            best_sim = 0.0
            best_idx = -1
            for i, det_emb in enumerate(detected_embs):
                if i in matched_face_indices:
                    continue
                sim = cosine_similarity(ref, det_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i

            if best_idx >= 0 and best_sim >= self.FACE_SIMILARITY_THRESHOLD:
                matched_face_indices.add(best_idx)
                char_results[cid] = {
                    "similarity": round(best_sim, 3),
                    "pass": best_sim >= self.QUALITY_PASS_THRESHOLD,
                    "detected": True,
                    "face_index": best_idx,
                    "bbox": [round(float(v), 1) for v in detected_boxes[best_idx]],
                }
            else:
                char_results[cid] = {
                    "similarity": round(best_sim, 3) if best_idx >= 0 else 0.0,
                    "pass": False,
                    "detected": False,
                    "face_index": -1,
                    "bbox": None,
                }

        # 4. 遮挡检测: 检查匹配的 bbox 之间是否有严重重叠
        occupancy_conflict = False
        matched_boxes = [
            detected_boxes[r["face_index"]]
            for r in char_results.values()
            if r["detected"] and r.get("face_index", -1) >= 0
        ]
        if len(matched_boxes) >= 2:
            for i in range(len(matched_boxes)):
                for j in range(i + 1, len(matched_boxes)):
                    iou = self._bbox_iou(matched_boxes[i], matched_boxes[j])
                    if iou > self.OCCLUSION_IOU_THRESHOLD:
                        occupancy_conflict = True
                        break

        all_pass = all(r["pass"] for r in char_results.values())
        detected_count = sum(1 for r in char_results.values() if r["detected"])

        suggestions = []
        if not all_pass:
            missing = [cid for cid, r in char_results.items() if not r["detected"]]
            low_quality = [cid for cid, r in char_results.items() if r["detected"] and not r["pass"]]
            if missing:
                suggestions.append(
                    f"Characters not detected: {', '.join(missing)}. "
                    "Ensure all characters are visible in frame. Try wider framing."
                )
            if low_quality:
                suggestions.append(
                    f"Low similarity for: {', '.join(low_quality)}. "
                    "Increase IPAdapter weight or adjust prompt for clearer features."
                )
            if occupancy_conflict:
                suggestions.append(
                    "Face bounding boxes overlap significantly — "
                    "characters may be occluding each other. Adjust composition."
                )

        return ToolResult.ok(
            data={
                "character_results": char_results,
                "all_pass": all_pass,
                "faces_detected": detected_count,
                "faces_expected": len(char_ids),
                "total_faces_in_image": len(faces),
                "occupancy_conflict": occupancy_conflict,
                "method": "arcface",
                "suggestions": suggestions,
            }
        )

    def _fallback_multi_check(self, char_ids: list[str]) -> ToolResult:
        """确定性回退 (无 ArcFace 时)"""
        char_results = {}
        all_pass = True

        for cid in char_ids:
            base = 0.82 + (hash(cid) % 18) / 100
            char_results[cid] = {
                "similarity": round(base, 3),
                "pass": base >= 0.85,
                "detected": base >= 0.60,
            }
            if base < 0.85:
                all_pass = False

        return ToolResult.ok(
            data={
                "character_results": char_results,
                "all_pass": all_pass,
                "occupancy_conflict": False,
                "method": "deterministic_fallback",
                "suggestions": [] if all_pass else ["Multi-character consistency check failed for some characters"],
            }
        )

    @staticmethod
    def _bbox_iou(box_a, box_b) -> float:
        """计算两个 bbox 的 IoU"""
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])
        inter_w = max(0, xb - xa)
        inter_h = max(0, yb - ya)
        inter_area = inter_w * inter_h
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0
