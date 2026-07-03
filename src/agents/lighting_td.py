"""灯光师 Agent — 镜头光照参数生成 + 场景内光照连续性守护"""

from __future__ import annotations

from src.core.agent import BaseAgent, AgentConfig
from src.data.protocols import LightingHint


class LightingTDAgent(BaseAgent):
    """
    职责:
      1. 为每个镜头生成精确的光照控制参数
      2. 检查相邻镜头的光照连续性
      3. 确保场景内光源方向、色温、强度一致

    关键: 把 StyleGuide 的抽象描述转成可量化的光照参数
    """

    def __init__(self, config: AgentConfig, bus, tools):
        super().__init__(config, bus, tools)
        self._scene_light_cache: dict[str, dict] = {}  # scene_id → base lighting

    async def handle_task(self, task: dict) -> dict:
        action = task.get("action", "generate")

        if action == "generate":
            return await self._generate_lighting(task)
        elif action == "check_continuity":
            return await self._check_continuity(task)
        elif action == "get_scene_base":
            return await self._get_scene_base(task)
        else:
            return {"status": "error", "error": f"Unknown action: {action}"}

    async def _generate_lighting(self, task: dict) -> dict:
        """为单个镜头生成光照参数"""
        shot = task.get("shot", {})
        style_guide = task.get("style_guide", {})
        scene_id = shot.get("scene_id", "")

        # 获取或创建场景基底光照
        scene_base = self._get_or_create_scene_base(scene_id, shot, style_guide)

        # 在基底上做镜头级别的微调
        shot_lighting = self._adapt_to_shot(scene_base, shot)

        await self.report("done", {"shot_id": shot.get("id", ""), "lighting": shot_lighting})

        return {
            "status": "ok",
            "lighting": shot_lighting,
        }

    async def _check_continuity(self, task: dict) -> dict:
        """检查前后镜头的光照连续性"""
        prev_hist = task.get("prev_histogram", [0.5] * 10)
        curr_hist = task.get("current_histogram", [0.5] * 10)
        scene_id = task.get("scene_id", "")

        result = await self.call_tool("light_continuity_checker", {
            "prev_histogram": prev_hist,
            "current_histogram": curr_hist,
        })

        data = result.data or {}
        passed = data.get("pass", True)

        if not passed:
            self.log.info(f"光照不连续: scene={scene_id}, drift={data.get('color_temp_drift', 0)}K")
            await self.report("continuity_failed", {
                "scene_id": scene_id,
                "drift": data,
            })

        return {
            "status": "ok",
            "pass": passed,
            "data": data,
        }

    async def _get_scene_base(self, task: dict) -> dict:
        scene_id = task.get("scene_id", "")
        return {
            "status": "ok",
            "base_lighting": self._scene_light_cache.get(scene_id, {}),
        }

    def _get_or_create_scene_base(self, scene_id: str, shot: dict, style_guide: dict) -> dict:
        """获取或初始化场景基底光照"""
        if scene_id in self._scene_light_cache:
            return self._scene_light_cache[scene_id]

        # 从 shot 和 style_guide 提取光照线索
        lighting_hint = shot.get("lighting_description", "")
        color_temp = shot.get("color_temp", 5600)

        visual_specs = style_guide.get("visual_specs", {})
        spec = visual_specs.get(scene_id, {})

        scene_mood = shot.get("emotion", "neutral")

        # 情绪 → 光照参数映射
        mood_lighting = {
            "loneliness": {"key_angle": [0.0, 0.3, -0.9], "intensity": 0.6, "fill": 0.15, "color": "#6B8DBF"},
            "tension": {"key_angle": [-0.5, 0.1, -0.7], "intensity": 0.9, "fill": 0.1, "color": "#4A6FA5"},
            "hope": {"key_angle": [0.0, 0.5, -0.3], "intensity": 0.85, "fill": 0.4, "color": "#F4D03F"},
            "sadness": {"key_angle": [0.3, 0.2, -0.5], "intensity": 0.5, "fill": 0.2, "color": "#7F8C8D"},
            "fear": {"key_angle": [-0.7, 0.05, -0.3], "intensity": 0.95, "fill": 0.05, "color": "#2C3E50"},
            "joy": {"key_angle": [0.0, 0.6, -0.4], "intensity": 0.8, "fill": 0.45, "color": "#FFD700"},
            "surprise": {"key_angle": [0.0, -0.2, -0.8], "intensity": 0.9, "fill": 0.35, "color": "#E8F8F5"},
            "wistful": {"key_angle": [0.2, 0.4, -0.6], "intensity": 0.65, "fill": 0.3, "color": "#D4A574"},
            "serene": {"key_angle": [0.0, 0.7, -0.2], "intensity": 0.7, "fill": 0.5, "color": "#FDEBD0"},
            "neutral": {"key_angle": [0.0, 0.5, -0.5], "intensity": 0.75, "fill": 0.3, "color": "#FFFFFF"},
            "anger": {"key_angle": [-0.8, 0.0, -0.4], "intensity": 1.0, "fill": 0.05, "color": "#E74C3C"},
        }

        mood_cfg = mood_lighting.get(scene_mood, mood_lighting["neutral"])

        base = {
            "scene_id": scene_id,
            "key_light": {
                "direction": mood_cfg["key_angle"],
                "color_temp_k": color_temp,
                "intensity": mood_cfg["intensity"],
                "color_hint": mood_cfg["color"],
            },
            "fill_light": {
                "intensity": mood_cfg["fill"],
                "color_temp_k": max(2800, color_temp - 800),
            },
            "rim_light": {
                "direction": [0.0, 0.1, -0.95],
                "color": spec.get("palette_accent", mood_cfg["color"]),
                "intensity": mood_cfg["intensity"] * 0.4,
            },
            "volumetrics": "dust_particles" if "废墟" in lighting_hint else "",
            "ambient_occlusion": 0.3,
        }

        self._scene_light_cache[scene_id] = base
        return base

    def _adapt_to_shot(self, scene_base: dict, shot: dict) -> dict:
        """在场景基底上做镜头级别微调"""
        import copy
        lighting = copy.deepcopy(scene_base)

        framing = shot.get("framing", "medium")
        movement = shot.get("camera_movement", "static")

        # 特写 → 降低填充光增强对比
        if framing in ["close_up", "extreme_close_up"]:
            lighting["fill_light"]["intensity"] *= 0.7
            lighting["key_light"]["intensity"] *= 1.05

        # 广角 → 提高填充光
        elif framing in ["wide", "extreme_wide"]:
            lighting["fill_light"]["intensity"] *= 1.2

        # 运镜 → 微调
        if movement in ["push_in", "tracking"]:
            lighting["key_light"]["direction"][0] += 0.05
        elif movement == "handheld":
            lighting["key_light"]["intensity"] *= 1.1

        # 对白镜头 → 面光增强
        if shot.get("dialog"):
            lighting["fill_light"]["intensity"] = max(lighting["fill_light"]["intensity"], 0.35)

        return lighting
