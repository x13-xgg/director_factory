"""分镜师 Agent — Screenplay → ShotList (系统的"编译器")"""

import uuid
from src.core.agent import BaseAgent, AgentConfig
from src.data.protocols import (
    ShotList, Shot, CameraSpec, Composition, Transition, LightingHint, AudioHint,
    Emotion, Framing, CameraAngle, CameraMovement, DepthOfField, TransitionType,
)


class StoryboarderAgent(BaseAgent):
    """
    核心定位: 系统的"编译器"——自然语言剧本 → 机器可执行的镜头指令
    """

    def __init__(self, config: AgentConfig, bus, tools):
        super().__init__(config, bus, tools)

    async def handle_task(self, task: dict) -> dict:
        screenplay = task.get("screenplay", {})

        self.log.info(f"开始编译分镜: {screenplay.get('title', 'Untitled')}")

        result = await self.call_tool("text_gen", {
            "system_prompt": self._build_system_prompt(),
            "user_prompt": self._build_user_prompt(screenplay),
            "output_schema": ShotListSchema,
            "temperature": 0.5,  # 较低温度，需要精确性
            "max_tokens": 8192,
        })

        data = result.data if isinstance(result.data, dict) else {}

        # 解析镜头列表
        shots = []
        for i, s in enumerate(data.get("shots", [])):
            shot = Shot(
                id=s.get("id", f"shot_{i:03d}"),
                scene_id=s.get("scene_id", ""),
                shot_number=i + 1,
                duration=s.get("duration", 3.0),
                camera=CameraSpec(
                    framing=Framing(s.get("framing", "medium")),
                    angle=CameraAngle(s.get("camera_angle", "eye_level")),
                    movement=CameraMovement(s.get("camera_movement", "static")),
                    depth_of_field=DepthOfField(s.get("depth_of_field", "medium")),
                ),
                composition=Composition(
                    subject=s.get("subject", ""),
                    position=s.get("composition_position", "center"),
                    background=s.get("background", ""),
                ),
                lighting=LightingHint(
                    description=s.get("lighting_description", ""),
                    color_temp_k=s.get("color_temp", 5600),
                ),
                audio=AudioHint(
                    ambience=s.get("ambience", ""),
                    foley=s.get("foley", ""),
                    bgm_mood=s.get("bgm_mood", ""),
                ),
                dialog=s.get("dialog", ""),
                emotion=Emotion(s.get("emotion", "neutral")),
                emotion_intensity=s.get("emotion_intensity", 0.5),
                transition_in=Transition(type=TransitionType(s.get("transition_in", "cut"))),
                transition_out=Transition(
                    type=TransitionType(s.get("transition_out", "cut")),
                    overlap=s.get("transition_overlap", 0.0),
                ),
                dependencies=s.get("dependencies", []),
                characters_in_frame=s.get("characters_in_frame", []),
                action_description=s.get("action", ""),
            )
            shots.append(shot)

        # 自动推断依赖关系
        shots = self._infer_dependencies(shots)

        shotlist = ShotList(
            project=screenplay.get("title", "Untitled"),
            total_duration=sum(s.duration for s in shots),
            shots=shots,
        )

        await self.report("done", {"shots": len(shots), "total_duration": shotlist.total_duration})

        return {
            "status": "ok",
            "shotlist": self._shotlist_to_dict(shotlist),
        }

    def _infer_dependencies(self, shots: list[Shot]) -> list[Shot]:
        """自动推断镜头间的依赖关系"""
        for i, shot in enumerate(shots):
            if i == 0:
                continue
            prev = shots[i - 1]
            # 同场景 + 同角色 → 可能存在前后依赖
            if shot.scene_id == prev.scene_id:
                if set(shot.characters_in_frame) & set(prev.characters_in_frame):
                    if prev.id not in shot.dependencies:
                        shot.dependencies.append(prev.id)
        return shots

    def _build_system_prompt(self) -> str:
        return """你是一位资深分镜师/故事板艺术家。你的任务是把剧本翻译为精确的、机器可执行的镜头指令。

对每个镜头，你必须指定:
- framing: extreme_wide / wide / medium_wide / medium / medium_close / close_up / extreme_close_up
- camera_angle: eye_level / low / high / dutch / overhead
- camera_movement: static / push_in / pull_out / pan_left / pan_right / tilt_up / tilt_down / tracking / handheld
- depth_of_field: shallow / medium / deep
- composition_position: center / center_third / left_third / right_third / upper_third / lower_third
- lighting_description: 简洁的光照描述 (如 "single source from window, soft shadows")
- color_temp: 色温 (3200=暖, 4400=中性, 5600=日光, 6500+=冷)
- emotion: joy / sadness / anger / fear / surprise / loneliness / hope / tension / serene / wistful / neutral
- emotion_intensity: 0.0-1.0
- transition_in / transition_out: cut / dissolve / fade / wipe / match
- 如果有对白，填入 dialog 字段
- 标注 characters_in_frame: 画面中的角色 ID 列表

原则:
1. 情感强度应延续或有机变化，不要突兀跳跃
2. 转场类型与情绪匹配 (快节奏→cut, 抒情→dissolve)
3. 镜头时长遵循"紧张短、舒缓长"的原则"""

    def _build_user_prompt(self, screenplay: dict) -> str:
        import json
        return f"""将以下剧本编译为分镜表 (ShotList):

剧本:
{json.dumps(screenplay, indent=2, ensure_ascii=False)}

要求:
- 每个场景拆分为 2-6 个镜头
- 为每个镜头指定完整的景别、机位、运镜、光照、情绪
- 镜头之间建立依赖关系 (后面的镜头如果依赖于前面的角色/场景状态)
- 总时长控制在 {screenplay.get('target_duration', 60)} 秒左右"""

    def _shotlist_to_dict(self, sl: ShotList) -> dict:
        return {
            "project": sl.project,
            "total_duration": sl.total_duration,
            "shots": [
                {
                    "id": s.id,
                    "scene_id": s.scene_id,
                    "shot_number": s.shot_number,
                    "duration": s.duration,
                    "framing": s.camera.framing.value,
                    "camera_angle": s.camera.angle.value,
                    "camera_movement": s.camera.movement.value,
                    "depth_of_field": s.camera.depth_of_field.value,
                    "subject": s.composition.subject,
                    "composition_position": s.composition.position,
                    "background": s.composition.background,
                    "lighting_description": s.lighting.description,
                    "color_temp": s.lighting.color_temp_k,
                    "ambience": s.audio.ambience,
                    "foley": s.audio.foley,
                    "bgm_mood": s.audio.bgm_mood,
                    "dialog": s.dialog,
                    "emotion": s.emotion.value,
                    "emotion_intensity": s.emotion_intensity,
                    "transition_in": s.transition_in.type.value,
                    "transition_out": s.transition_out.type.value,
                    "transition_overlap": s.transition_out.overlap,
                    "dependencies": s.dependencies,
                    "characters_in_frame": s.characters_in_frame,
                    "action": s.action_description,
                }
                for s in sl.shots
            ],
        }


ShotListSchema = {
    "type": "object",
    "properties": {
        "shots": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "scene_id": {"type": "string"},
                    "duration": {"type": "number"},
                    "framing": {"type": "string", "enum": [e.value for e in Framing]},
                    "camera_angle": {"type": "string", "enum": [e.value for e in CameraAngle]},
                    "camera_movement": {"type": "string", "enum": [e.value for e in CameraMovement]},
                    "depth_of_field": {"type": "string", "enum": [e.value for e in DepthOfField]},
                    "subject": {"type": "string"},
                    "composition_position": {"type": "string"},
                    "background": {"type": "string"},
                    "lighting_description": {"type": "string"},
                    "color_temp": {"type": "integer"},
                    "ambience": {"type": "string"},
                    "foley": {"type": "string"},
                    "bgm_mood": {"type": "string"},
                    "dialog": {"type": "string"},
                    "emotion": {"type": "string", "enum": [e.value for e in Emotion]},
                    "emotion_intensity": {"type": "number"},
                    "transition_in": {"type": "string", "enum": [e.value for e in TransitionType]},
                    "transition_out": {"type": "string", "enum": [e.value for e in TransitionType]},
                    "transition_overlap": {"type": "number"},
                    "dependencies": {"type": "array", "items": {"type": "string"}},
                    "characters_in_frame": {"type": "array", "items": {"type": "string"}},
                    "action": {"type": "string"},
                },
                "required": ["id", "duration", "framing", "camera_angle", "camera_movement", "emotion", "emotion_intensity"],
            },
        },
    },
    "required": ["shots"],
}
