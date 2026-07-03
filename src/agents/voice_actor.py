"""配音演员 Agent — TTS 对白生成 + 情绪表达控制"""

from __future__ import annotations

import asyncio
from pathlib import Path

from src.core.agent import BaseAgent, AgentConfig
from src.data.protocols import AudioClip, Emotion


VOICE_PROFILES = {
    "机械但温柔, 略带杂音": {"speed": 0.92, "pitch_variance": 0.15, "profile": "robot_warm"},
    "冰冷, 精确, 不带感情": {"speed": 1.05, "pitch_variance": 0.02, "profile": "robot_cold"},
    "低沉沙哑": {"speed": 0.85, "pitch_variance": 0.08, "profile": "gravelly"},
    "轻快明亮": {"speed": 1.1, "pitch_variance": 0.12, "profile": "bright"},
    "低沉威胁": {"speed": 0.78, "pitch_variance": 0.05, "profile": "menacing"},
}

EMOTION_SPEED_MAP = {
    "fear": 0.88, "anger": 1.15, "sadness": 0.82, "joy": 1.10,
    "tension": 0.95, "loneliness": 0.80, "hope": 1.0, "surprise": 1.20,
    "wistful": 0.85, "serene": 0.90, "neutral": 1.0,
}

EMOTION_PITCH_MAP = {
    "fear": 0.18, "anger": 0.25, "sadness": -0.10, "joy": 0.20,
    "tension": 0.12, "loneliness": -0.08, "hope": 0.08, "surprise": 0.30,
    "wistful": -0.05, "serene": -0.03, "neutral": 0.0,
}


class VoiceActorAgent(BaseAgent):
    """
    职责:
      1. 为有对白的镜头生成 TTS 配音
      2. 根据角色 voice_style 选择合适的音色配置
      3. 根据镜头 emotion 调整语速和音调变化
      4. 生成带时间戳的配音片段，供后期精确对齐
    """

    def __init__(self, config: AgentConfig, bus, tools):
        super().__init__(config, bus, tools)
        self._clip_cache: dict[str, AudioClip] = {}

    async def handle_task(self, task: dict) -> dict:
        action = task.get("action", "generate")

        if action == "generate":
            return await self._generate_dialog(task)
        elif action == "generate_all":
            return await self._generate_all(task)
        elif action == "get_clip":
            return await self._get_clip(task)
        else:
            return {"status": "error", "error": f"Unknown action: {action}"}

    async def _generate_dialog(self, task: dict) -> dict:
        """为单个镜头生成对白配音"""
        shot = task.get("shot", {})
        shot_id = shot.get("id", "")
        dialog = shot.get("dialog", "")
        emotion = shot.get("emotion", "neutral")
        characters_in_frame = shot.get("characters_in_frame", [])
        char_profiles = task.get("char_profiles", {})
        language = task.get("language", "zh")

        if not dialog:
            return {
                "status": "ok",
                "shot_id": shot_id,
                "audio_clip": None,
                "note": "no dialog to generate",
            }

        # 确定主要说话人 (取帧内第一个角色)
        speaker_id = characters_in_frame[0] if characters_in_frame else "narrator"
        speaker_profile = char_profiles.get(speaker_id, {})
        voice_style = speaker_profile.get("voice_style", "") if isinstance(speaker_profile, dict) else ""

        # 音色调配
        voice_cfg = VOICE_PROFILES.get(voice_style, {"speed": 1.0, "pitch_variance": 0.0, "profile": "default"})
        speed_base = voice_cfg["speed"]
        pitch_base = voice_cfg["pitch_variance"]

        # 情绪调整
        emo_speed = EMOTION_SPEED_MAP.get(emotion, 1.0)
        emo_pitch = EMOTION_PITCH_MAP.get(emotion, 0.0)
        final_speed = speed_base * emo_speed
        final_pitch = max(-0.3, min(0.4, pitch_base + emo_pitch))

        # 调用 TTS 工具
        result = await self.call_tool("tts", {
            "text": dialog,
            "emotion": emotion,
            "character_id": speaker_id,
            "voice_profile": voice_cfg["profile"],
            "speed": round(final_speed, 2),
            "pitch_variance": round(final_pitch, 2),
            "language": language,
        })

        data = result.data or {}

        audio_clip = AudioClip(
            shot_id=shot_id,
            audio_path=data.get("audio_path", ""),
            text=dialog,
            emotion=Emotion(emotion) if emotion in Emotion._value2member_map_ else Emotion.NEUTRAL,
            duration=data.get("duration", 0.0),
            phoneme_timestamps=data.get("phoneme_timestamps", []),
        )

        self._clip_cache[shot_id] = audio_clip

        self.log.info(
            f"配音生成: shot={shot_id}, speaker={speaker_id}, "
            f"duration={audio_clip.duration:.1f}s, emotion={emotion}, "
            f"speed={final_speed:.2f}, pitch={final_pitch:.2f}"
        )

        await self.report("done", {
            "shot_id": shot_id,
            "audio_path": audio_clip.audio_path,
            "duration": audio_clip.duration,
        })

        return {
            "status": "ok",
            "shot_id": shot_id,
            "audio_clip": self._clip_to_dict(audio_clip),
        }

    async def _generate_all(self, task: dict) -> dict:
        """批量为所有有对白的镜头生成配音"""
        shots = task.get("shots", [])
        char_profiles = task.get("char_profiles", {})
        language = task.get("language", "zh")

        clips = {}
        for shot in shots:
            if not shot.get("dialog"):
                continue
            result = await self._generate_dialog({
                "shot": shot,
                "char_profiles": char_profiles,
                "language": language,
            })
            if result.get("audio_clip"):
                clips[shot.get("id", "")] = result["audio_clip"]

        self.log.info(f"批量配音完成: {len(clips)} 个片段")

        return {
            "status": "ok",
            "clips": clips,
            "count": len(clips),
        }

    async def _get_clip(self, task: dict) -> dict:
        shot_id = task.get("shot_id", "")
        clip = self._clip_cache.get(shot_id)
        if clip:
            return {"status": "ok", "audio_clip": self._clip_to_dict(clip)}
        return {"status": "ok", "audio_clip": None}

    def _clip_to_dict(self, clip: AudioClip) -> dict:
        return {
            "shot_id": clip.shot_id,
            "audio_path": clip.audio_path,
            "text": clip.text,
            "emotion": clip.emotion.value if isinstance(clip.emotion, Emotion) else clip.emotion,
            "duration": clip.duration,
            "phoneme_timestamps": clip.phoneme_timestamps,
        }
