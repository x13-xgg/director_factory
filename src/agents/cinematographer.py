"""摄影师 Agent — Shot 分镜指令 → Frame 画面帧 (含自检回路)"""

import random
from pathlib import Path

from src.core.agent import BaseAgent, AgentConfig
from src.data.protocols import Frame


class CinematographerAgent(BaseAgent):
    """
    职责: 把分镜师的镜头指令翻译为图像/视频
    只为单个镜头负责，不关心前后镜头关系

    流程:
      1. 组装 prompt (镜头描述 + 风格约束)
      2. 生成画面 (mock or real SD/ComfyUI)
      3. 自检 composition_scorer
      4. 不通过 → 微调参数重新生成 (最多 retry 3 次)
    """

    def __init__(self, config: AgentConfig, bus, tools):
        super().__init__(config, bus, tools)
        self.image_counter: dict[str, int] = {}  # 追踪每个镜头的重试次数

    async def handle_task(self, task: dict) -> dict:
        shot = task.get("shot", {})
        style_guide = task.get("style_guide", {})
        char_profiles = task.get("char_profiles", {})

        shot_id = shot.get("id", "unknown")
        self.image_counter[shot_id] = 0

        return await self._shoot_with_retry(shot, style_guide, char_profiles)

    async def _shoot_with_retry(self, shot: dict, style_guide: dict, char_profiles: dict) -> dict:
        shot_id = shot.get("id", "unknown")
        max_retries = self.config.max_retries

        for attempt in range(max_retries):
            self.image_counter[shot_id] = attempt + 1

            # Step 1: 组装 prompt
            prompt, negative = self._build_prompt(shot, style_guide, char_profiles)

            # Step 2: 生成画面
            gen_result = await self._generate_frame(shot, prompt, negative)

            # Step 3: 自检 (composition)
            score_result = await self.call_tool("composition_scorer", {
                "image_path": gen_result.get("image_path", ""),
                "shot_spec": shot,
            })

            comp_score = score_result.data.get("overall", 0.85) if score_result.data else 0.85

            # Step 3b: 多角色同框检测
            chars_in_frame = shot.get("characters_in_frame", [])
            multi_char_pass = True
            multi_char_data = None
            if len(chars_in_frame) >= 2:
                mc_result = await self.call_tool("multi_char_composition", {
                    "generated_image_path": gen_result.get("image_path", ""),
                    "character_ids": chars_in_frame,
                })
                multi_char_data = mc_result.data
                multi_char_pass = multi_char_data.get("all_pass", True) if multi_char_data else True
                if not multi_char_pass:
                    self.log.info(
                        f"镜头 {shot_id} 多角色一致性不通过: {multi_char_data.get('suggestions', [])}"
                    )

            # Step 4: 判定
            if comp_score >= 0.82 and multi_char_pass:
                frame = Frame(
                    shot_id=shot_id,
                    image_path=gen_result.get("image_path", ""),
                    prompt=prompt,
                    negative_prompt=negative,
                    seed=gen_result.get("seed", 0),
                    composition_score=comp_score,
                    metadata={
                        "attempt": attempt + 1,
                        "gen_method": gen_result.get("method", "mock"),
                        "scores": score_result.data,
                        "multi_char_check": multi_char_data,
                    },
                )
                await self.report("shot_done", {"shot_id": shot_id, "score": comp_score, "attempts": attempt + 1})
                return {
                    "status": "ok",
                    "frame": self._frame_to_dict(frame),
                }
            else:
                suggestions = list(score_result.data.get("suggestions", []) if score_result.data else [])
                if not multi_char_pass and multi_char_data:
                    suggestions.extend(multi_char_data.get("suggestions", []))
                self.log.info(f"镜头 {shot_id} 自检不通过 (comp={comp_score:.2f}, multi_char={multi_char_pass})，调整重试... {suggestions}")
                shot = self._adjust_shot(shot, suggestions)

        # 达到最大重试 → 返回当前最佳
        frame = Frame(
            shot_id=shot_id,
            image_path=gen_result.get("image_path", ""),
            prompt=prompt,
            negative_prompt=negative,
            seed=gen_result.get("seed", 0),
            composition_score=comp_score,
            metadata={
                "attempt": max_retries,
                "exhausted_retries": True,
                "multi_char_check": multi_char_data,
            },
        )
        await self.report("shot_failed", {"shot_id": shot_id, "score": comp_score})
        return {"status": "warning", "frame": self._frame_to_dict(frame), "warning": "max retries exhausted"}

    def _build_prompt(self, shot: dict, style_guide: dict, char_profiles: dict) -> tuple[str, str]:
        """组装 Stable Diffusion prompt"""
        parts = []

        # 景别
        framing = shot.get("framing", "medium")
        framing_prompts = {
            "extreme_wide": "extreme wide shot, vast landscape, tiny subject",
            "wide": "wide shot, full body in frame, establishing",
            "medium_wide": "medium wide shot, three-quarter body",
            "medium": "medium shot, waist up",
            "medium_close": "medium close-up, chest up",
            "close_up": "close-up, face filling frame",
            "extreme_close_up": "extreme close-up, macro detail shot",
        }
        parts.append(framing_prompts.get(framing, "medium shot"))

        # 主体
        subject = shot.get("subject", "")
        if subject:
            parts.append(subject)

        # 构图
        pos = shot.get("composition_position", "center")
        parts.append(f"subject positioned at {pos}")

        # 背景
        bg = shot.get("background", "")
        if bg:
            parts.append(f"background: {bg}")

        # 运镜
        movement = shot.get("camera_movement", "static")
        if movement != "static":
            parts.append(f"camera {movement.replace('_', ' ')}")

        # 景深
        dof = shot.get("depth_of_field", "medium")
        if dof == "shallow":
            parts.append("shallow depth of field, bokeh background")
        elif dof == "deep":
            parts.append("deep depth of field, everything in focus")

        # 光照
        light = shot.get("lighting_description", "")
        if light:
            parts.append(f"lighting: {light}")

        # 场景风格
        scene_id = shot.get("scene_id", "")
        visual_specs = style_guide.get("visual_specs", {})
        spec = visual_specs.get(scene_id, {})
        texture = spec.get("texture_prompt", "")
        if texture:
            parts.append(texture)

        # 角色引用 (IPAdapter trigger)
        chars = shot.get("characters_in_frame", [])
        for cid in chars:
            if cid in char_profiles:
                parts.append(char_profiles[cid].get("base_prompt", ""))

        # 全局风格
        parts.append("cinematic lighting, photorealistic, 8k, high detail")

        prompt = ", ".join(filter(None, parts))

        # 负面 prompt
        neg_parts = ["cartoon", "3d render", "anime", "low quality", "blurry",
                      "deformed", "disfigured", "bad anatomy", "watermark", "text"]
        global_neg = spec.get("negative_prompt", "")
        if global_neg:
            neg_parts.extend(global_neg.split(", "))
        negative = ", ".join(neg_parts)

        return prompt, negative

    async def _generate_frame(self, shot: dict, prompt: str, negative: str) -> dict:
        """生成画面 — mock 或 ComfyUI SDXL"""

        shot_id = shot.get("id", "unknown")
        output_dir = Path("outputs/frames")
        output_dir.mkdir(parents=True, exist_ok=True)
        attempt = self.image_counter.get(shot_id, 1)
        filename = f"{shot_id}_v{attempt}.png"

        # 尝试调用 ComfyUI image_gen 工具
        gen_result = await self.tools.call("image_gen", {
            "prompt": prompt,
            "negative_prompt": negative,
            "width": self._get_width(shot),
            "height": self._get_height(shot),
            "steps": shot.get("steps", 4),
            "cfg": shot.get("cfg", 2.0),
            "output_dir": str(output_dir),
            "filename": filename,
        })

        if gen_result.status == "ok" and gen_result.data:
            images = gen_result.data.get("images", [])
            if images and Path(images[0]).exists() and Path(images[0]).stat().st_size > 100:
                return {
                    "image_path": images[0],
                    "seed": gen_result.data.get("seed", 0),
                    "method": gen_result.data.get("gen_method", "comfyui"),
                    "prompt": prompt,
                }

        # Fallback: mock
        seed = random.randint(1, 2_147_483_647)
        image_path = str(output_dir / filename)
        Path(image_path).touch()

        return {
            "image_path": image_path,
            "seed": seed,
            "method": "mock",
            "prompt": prompt,
        }

    def _get_width(self, shot: dict) -> int:
        from src.core.config import config as cfg
        framing = shot.get("framing", "medium")
        w = cfg.image_gen.default_width
        return w

    def _get_height(self, shot: dict) -> int:
        from src.core.config import config as cfg
        return cfg.image_gen.default_height

    def _adjust_shot(self, shot: dict, suggestions: list[str]) -> dict:
        """根据评分建议微调镜头参数"""
        adjusted = dict(shot)
        for sug in suggestions:
            sug_lower = sug.lower()
            if "controlnet" in sug_lower:
                # 增强 controlnet 强度
                adjusted["_controlnet_strength"] = shot.get("_controlnet_strength", 1.0) * 1.15
            if "深度" in sug_lower or "depth" in sug_lower:
                adjusted["depth_of_field"] = "shallow"
            if "构图" in sug_lower or "composition" in sug_lower:
                adjusted["composition_position"] = "center_third"
            if "对比" in sug_lower or "contrast" in sug_lower:
                current_light = adjusted.get("lighting_description", "")
                adjusted["lighting_description"] = current_light + ", high contrast"
        return adjusted

    def _frame_to_dict(self, f: Frame) -> dict:
        return {
            "shot_id": f.shot_id,
            "image_path": f.image_path,
            "prompt": f.prompt,
            "seed": f.seed,
            "composition_score": f.composition_score,
            "metadata": f.metadata,
        }
