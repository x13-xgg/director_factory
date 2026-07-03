"""美术指导 Agent — ShotList + StyleGuide → VisualSpec"""

from src.core.agent import BaseAgent, AgentConfig
from src.data.protocols import StyleGuide, VisualSpec


class ArtDirectorAgent(BaseAgent):
    """
    职责: 为每场戏生成统一的视觉参数
    不生成画面，只提供"风格的约束条件"
    """

    def __init__(self, config: AgentConfig, bus, tools):
        super().__init__(config, bus, tools)

    async def handle_task(self, task: dict) -> dict:
        shotlist = task.get("shotlist", {})
        style_hint = task.get("style_hint", "")

        self.log.info("开始设计视觉风格方案")

        # 收集所有场景 ID
        scene_ids = list({s["scene_id"] for s in shotlist.get("shots", []) if s.get("scene_id")})

        result = await self.call_tool("text_gen", {
            "system_prompt": self._build_system_prompt(),
            "user_prompt": self._build_user_prompt(shotlist, style_hint),
            "output_schema": StyleGuideSchema,
            "temperature": 0.6,
            "max_tokens": 4096,
        })

        data = result.data if isinstance(result.data, dict) else {}
        scene_moods = self._extract_moods(shotlist)

        visual_specs = {}
        for item in data.get("visual_specs", []):
            sid = item.get("scene_id", "")
            visual_specs[sid] = VisualSpec(
                scene_id=sid,
                palette_dominant=item.get("palette_dominant", ""),
                palette_accent=item.get("palette_accent", ""),
                mood_descriptor=scene_moods.get(sid, ""),
                texture_prompt=item.get("texture_prompt", ""),
                negative_prompt=item.get("negative_prompt", ""),
                reference_style=item.get("reference_style", ""),
            )

        # 为未覆盖的场景生成默认 visual spec
        for sid in scene_ids:
            if sid not in visual_specs:
                visual_specs[sid] = VisualSpec(
                    scene_id=sid,
                    palette_dominant=data.get("global_palette", "#2C3E50"),
                    palette_accent=data.get("accent_color", "#E74C3C"),
                    mood_descriptor=scene_moods.get(sid, "neutral"),
                    texture_prompt=data.get("texture_style", "photorealistic, cinematic"),
                    negative_prompt=data.get("global_negative", "cartoon, 3d render, anime"),
                )

        style_guide = StyleGuide(
            project=shotlist.get("project", "Untitled"),
            global_palette=data.get("global_palette", ""),
            lighting_style=data.get("lighting_style", ""),
            visual_mood=data.get("visual_mood", ""),
            visual_specs={k: self._vs_to_dict(v) for k, v in visual_specs.items()},
        )

        await self.report("done", {"scenes_with_specs": len(visual_specs)})

        return {
            "status": "ok",
            "style_guide": self._sg_to_dict(style_guide),
        }

    def _extract_moods(self, shotlist: dict) -> dict[str, str]:
        moods = {}
        for s in shotlist.get("shots", []):
            sid = s.get("scene_id")
            if sid and sid not in moods:
                moods[sid] = s.get("emotion", "neutral")
        return moods

    def _build_system_prompt(self) -> str:
        return """你是一位电影美术指导/调色师。你的任务是为一部短片设计统一的视觉风格方案。

对每个场景，指定:
1. 主色调 (palette_dominant) — 十六进制颜色
2. 强调色 (palette_accent) — 用于视觉焦点
3. 质感描述 (texture_prompt) — 用于图像生成的正面提示词
4. 排除元素 (negative_prompt) — 不希望出现的视觉元素
5. 参考风格 (reference_style) — 如 "Blade Runner meets Wall-E"

原则: 全片色调统一但有场景间的有机变化。"""

    def _build_user_prompt(self, shotlist: dict, style_hint: str) -> str:
        import json
        moods = self._extract_moods(shotlist)
        return f"""根据以下分镜表和场景情绪设计视觉风格:

场景情绪: {json.dumps(moods, ensure_ascii=False)}
风格提示: {style_hint or 'cinematic, photorealistic'}
项目名: {shotlist.get('project', 'Untitled')}

请为每个场景设计调色板、质感和风格参考。"""

    def _vs_to_dict(self, vs: VisualSpec) -> dict:
        return {
            "scene_id": vs.scene_id,
            "palette_dominant": vs.palette_dominant,
            "palette_accent": vs.palette_accent,
            "mood_descriptor": vs.mood_descriptor,
            "texture_prompt": vs.texture_prompt,
            "negative_prompt": vs.negative_prompt,
            "reference_style": vs.reference_style,
        }

    def _sg_to_dict(self, sg: StyleGuide) -> dict:
        return {
            "project": sg.project,
            "global_palette": sg.global_palette,
            "lighting_style": sg.lighting_style,
            "visual_mood": sg.visual_mood,
            "visual_specs": sg.visual_specs,
        }


StyleGuideSchema = {
    "type": "object",
    "properties": {
        "global_palette": {"type": "string", "description": "全片主色调十六进制"},
        "accent_color": {"type": "string", "description": "全片强调色"},
        "lighting_style": {"type": "string", "description": "光照风格描述"},
        "visual_mood": {"type": "string", "description": "全片视觉情绪"},
        "texture_style": {"type": "string", "description": "默认质感描述"},
        "global_negative": {"type": "string", "description": "全局负面提示词"},
        "visual_specs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scene_id": {"type": "string"},
                    "palette_dominant": {"type": "string"},
                    "palette_accent": {"type": "string"},
                    "texture_prompt": {"type": "string"},
                    "negative_prompt": {"type": "string"},
                    "reference_style": {"type": "string"},
                },
                "required": ["scene_id", "palette_dominant", "texture_prompt", "negative_prompt"],
            },
        },
    },
    "required": ["global_palette", "visual_specs"],
}
