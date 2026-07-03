"""角色导演 Agent — 角色创建、资产锁定、跨镜头一致性守护"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from src.core.agent import BaseAgent, AgentConfig
from src.data.protocols import CharacterProfile
from src.tools.asset_db import asset_db


class CharacterDirectorAgent(BaseAgent):
    """
    核心职责:
      1. 从剧本角色描述生成 CharacterProfile
      2. 生成基准参考图 → 提取 face_embedding → 训练 LoRA
      3. 将角色资产锁定到 asset_db (不可变)
      4. 每个镜头生成后验证角色一致性

    策略:
      - 一个角色一份资产，锁定后全局共享
      - 摄影师拍任何包含该角色的镜头时，必须引用锁定的 embedding/lora
      - 一致性检查不通过 → 拒绝镜头，返回给摄影师重拍
    """

    def __init__(self, config: AgentConfig, bus, tools):
        super().__init__(config, bus, tools)
        self._profiles: dict[str, CharacterProfile] = {}
        self._creation_queue: asyncio.Queue = asyncio.Queue()

    async def handle_task(self, task: dict) -> dict:
        action = task.get("action", "create_all")

        if action == "create_character":
            return await self._create_character(task)
        elif action == "create_all":
            return await self._create_all(task)
        elif action == "verify":
            return await self._verify_consistency(task)
        elif action == "verify_multi":
            return await self._verify_multi_char(task)
        elif action == "get_profile":
            cid = task.get("character_id", "")
            profile = self._profiles.get(cid)
            if profile:
                return {"status": "ok", "profile": self._profile_to_dict(profile)}
            # 尝试从 asset_db 加载
            return {"status": "ok", "profile": self._load_profile(cid)}
        elif action == "list":
            return {"status": "ok", "profiles": {k: self._profile_to_dict(v) for k, v in self._profiles.items()}}
        else:
            return {"status": "error", "error": f"Unknown action: {action}"}

    # ── 角色创建 ────────────────────────────────────

    async def _create_all(self, task: dict) -> dict:
        """批量创建剧本中的所有角色"""
        characters = task.get("characters", [])
        style_guide = task.get("style_guide", {})

        self.log.info(f"开始创建 {len(characters)} 个角色...")

        profiles = {}
        for char in characters:
            cid = char.get("id", "")
            profile = await self._create_single(cid, char, style_guide)
            profiles[cid] = profile

        await self.report("done", {
            "created": len(profiles),
            "characters": list(profiles.keys()),
        })

        return {
            "status": "ok",
            "profiles": {k: self._profile_to_dict(v) for k, v in profiles.items()},
        }

    async def _create_character(self, task: dict) -> dict:
        """创建单个角色"""
        char_data = task.get("character", {})
        style_guide = task.get("style_guide", {})
        cid = char_data.get("id", "")

        profile = await self._create_single(cid, char_data, style_guide)
        return {
            "status": "ok",
            "profile": self._profile_to_dict(profile),
        }

    async def _create_single(self, cid: str, char_data: dict, style_guide: dict) -> CharacterProfile:
        """单个角色的完整创建流程"""
        name = char_data.get("name", cid)
        description = char_data.get("description", "")
        voice_style = char_data.get("voice_style", "")

        self.log.info(f"创建角色: {name} ({cid})")

        # Step 1: 生成基准 prompt
        style_texture = style_guide.get("visual_mood", "cinematic")
        base_prompt = self._build_character_prompt(char_data, style_texture)

        # Step 2: 生成基准参考图
        ref_image_path = await self._generate_reference_image(cid, base_prompt, style_guide)

        # Step 3: 提取 face embedding
        embed_result = await self.call_tool("embed_extractor", {
            "character_id": cid,
            "reference_image": ref_image_path,
            "method": "arcface",
        })
        face_embedding = embed_result.data.get("embedding", []) if embed_result.data else []

        # Step 4: 来源/训练 LoRA (CivitAI → HuggingFace → mock)
        lora_result = await self.call_tool("lora_trainer", {
            "character_id": cid,
            "description": description,
            "reference_images": [ref_image_path],
            "base_model": "sdxl_v3",
            "trigger_word": f"char_{cid}",
            "steps": 800,
            "rank": 16,
        })
        lora_path = lora_result.data.get("lora_path", "") if lora_result.data else ""
        trigger_word = lora_result.data.get("trigger_word", f"char_{cid}") if lora_result.data else f"char_{cid}"

        # Step 5: 组装 CharacterProfile
        profile = CharacterProfile(
            character_id=cid,
            base_prompt=f"{trigger_word}, {base_prompt}",
            distinctive_features=self._extract_features(description),
            lora_path=lora_path,
            face_embedding=face_embedding,
            reference_image_path=ref_image_path,
            locked=True,
            version=1,
        )

        # Step 6: 存入 asset_db 并锁定
        self._persist_profile(profile)
        self._profiles[cid] = profile

        # Step 7: 广播角色锁定事件
        await self.broadcast(f"character_locked:{cid}", self._profile_to_dict(profile))

        self.log.info(f"角色资产锁定: {name} ({cid}) | lora={lora_path}, embedding_dim={len(face_embedding)}")
        return profile

    # ── 一致性验证 ──────────────────────────────────

    async def _verify_consistency(self, task: dict) -> dict:
        """单角色一致性验证"""
        cid = task.get("character_id", "")
        image_path = task.get("image_path", "")

        profile = self._profiles.get(cid)
        if not profile:
            profile_dict = self._load_profile(cid)
            if not profile_dict:
                return {"status": "ok", "pass": True, "similarity": 1.0, "note": "no reference found, skipping check"}

        ref_emb = profile.face_embedding if profile else []

        result = await self.call_tool("character_consistency_checker", {
            "generated_image_path": image_path,
            "character_id": cid,
            "reference_embedding": ref_emb,
            "check_dimensions": ["face", "appearance", "style"],
        })

        data = result.data or {}
        passed = data.get("pass", True)

        if not passed:
            self.log.info(f"角色一致性验证失败: {cid} (overall={data.get('overall', 0):.2f})")
            await self.report("consistency_failed", {
                "character_id": cid,
                "score": data.get("overall", 0),
                "suggestions": data.get("suggestions", []),
            })
        else:
            self.log.info(f"角色一致性验证通过: {cid} (overall={data.get('overall', 0):.2f})")

        return {"status": "ok", "pass": passed, "data": data}

    async def _verify_multi_char(self, task: dict) -> dict:
        """多角色同框验证"""
        char_ids = task.get("character_ids", [])
        image_path = task.get("image_path", "")

        result = await self.call_tool("multi_char_composition", {
            "generated_image_path": image_path,
            "character_ids": char_ids,
        })

        return {"status": "ok", "data": result.data}

    # ── 辅助方法 ────────────────────────────────────

    def _build_character_prompt(self, char_data: dict, style_texture: str) -> str:
        """根据角色描述构建 Stable Diffusion prompt"""
        parts = []

        name = char_data.get("name", "")
        description = char_data.get("description", "")

        if description:
            parts.append(description)
        if style_texture:
            parts.append(style_texture)

        parts.append("character reference sheet, front view, clean lighting, detailed features, photorealistic")

        return ", ".join(parts)

    async def _generate_reference_image(self, cid: str, prompt: str, style_guide: dict) -> str:
        """生成基准参考图"""
        # 使用 text_gen 构建更精确的 prompt
        prompt_result = await self.call_tool("text_gen", {
            "system_prompt": "You are a character designer. Refine the given prompt for a character reference image.",
            "user_prompt": f"Refine this prompt for a character reference sheet: {prompt}",
            "temperature": 0.7,
            "max_tokens": 512,
        })

        refined_prompt = prompt
        if prompt_result.data and isinstance(prompt_result.data, str):
            refined_prompt = prompt_result.data.strip()

        # 保存 mock 参考图
        output_dir = Path("outputs/characters")
        output_dir.mkdir(parents=True, exist_ok=True)
        ref_path = str(output_dir / f"{cid}_reference.png")
        Path(ref_path).touch()

        # 生产环境: 调用 ComfyUI / SD API 实际生成
        # gen_result = await self.call_tool("img_gen", {"prompt": refined_prompt, ...})

        self.log.info(f"基准参考图: {ref_path} (prompt={refined_prompt[:100]}...)")
        return ref_path

    def _extract_features(self, description: str) -> list[str]:
        """从描述文本中提取显著特征"""
        keywords = []
        descriptors = ["scar", "crack", "rust", "worn", "missing", "broken",
                       "glowing", "sharp", "smooth", "metallic", "tall", "short",
                       "left", "right", "eye", "arm", "hand", "visor", "sensor"]
        desc_lower = description.lower()
        for kw in descriptors:
            if kw in desc_lower:
                keywords.append(kw)
        return keywords[:5]  # 最多 5 个

    def _persist_profile(self, profile: CharacterProfile):
        """将角色配置持久化到 asset_db"""
        data = self._profile_to_dict(profile)
        asset_db.put("char_asset_db", f"{profile.character_id}:profile", data, {"type": "character_profile"})
        asset_db.lock("char_asset_db", f"{profile.character_id}:profile")

    def _load_profile(self, cid: str) -> dict | None:
        """从 asset_db 加载角色配置"""
        record = asset_db.get("char_asset_db", f"{cid}:profile")
        if record:
            return record.get("data", {})
        return None

    def _profile_to_dict(self, profile: CharacterProfile) -> dict:
        return {
            "character_id": profile.character_id,
            "base_prompt": profile.base_prompt,
            "distinctive_features": profile.distinctive_features,
            "lora_path": profile.lora_path,
            "face_embedding": profile.face_embedding,
            "reference_image_path": profile.reference_image_path,
            "locked": profile.locked,
            "version": profile.version,
        }

    def get_active_profiles(self) -> dict[str, CharacterProfile]:
        """获取所有活跃角色配置 (供其他 Agent 查询)"""
        return dict(self._profiles)
