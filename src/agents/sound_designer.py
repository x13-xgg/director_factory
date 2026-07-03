"""音效师 Agent — SFX 匹配 + BGM 选择 + 混音"""

from __future__ import annotations

from src.core.agent import BaseAgent, AgentConfig


class SoundDesignerAgent(BaseAgent):
    """
    职责:
      1. 根据镜头 action_description + location 匹配音效
      2. 根据场景 emotion / mood 选择 BGM
      3. 混音: 对白 + SFX + BGM → 最终音轨
      4. 确保场景内音效连续性 (reverb / ambience 一致)
    """

    def __init__(self, config: AgentConfig, bus, tools):
        super().__init__(config, bus, tools)
        self._scene_ambience: dict[str, dict] = {}
        self._bgm_cache: dict[str, str] = {}

    async def handle_task(self, task: dict) -> dict:
        action = task.get("action", "design")

        if action == "design":
            return await self._design_soundscape(task)
        elif action == "design_scene":
            return await self._design_scene(task)
        elif action == "select_bgm":
            return await self._select_bgm(task)
        elif action == "mix":
            return await self._mix(task)
        else:
            return {"status": "error", "error": f"Unknown action: {action}"}

    async def _design_soundscape(self, task: dict) -> dict:
        """为单个镜头设计音效层"""
        shot = task.get("shot", {})
        shot_id = shot.get("id", "")
        action_desc = shot.get("action_description", "")
        location = shot.get("lighting", {})
        if isinstance(location, dict):
            location = location.get("description", "")
        shot_emotion = shot.get("emotion", "neutral")

        # 匹配 SFX
        sfx_result = await self.call_tool("sfx_matcher", {
            "action_description": action_desc,
            "location": str(location),
            "mood": shot_emotion,
            "max_results": 4,
        })
        sfx_data = sfx_result.data or {}
        sfx_matches = sfx_data.get("sfx_matches", [])

        # 获取或创建场景环境音
        scene_id = shot.get("scene_id", "")
        ambience = self._get_or_create_ambience(scene_id, shot, task.get("style_guide", {}))

        audio_hints = {
            "shot_id": shot_id,
            "scene_id": scene_id,
            "foley": [m["path"] for m in sfx_matches if m.get("confidence", 0) > 0.7],
            "foley_details": sfx_matches,
            "ambience": ambience.get("path", ""),
            "ambience_volume": ambience.get("volume", 0.3),
            "reverb": ambience.get("reverb", "medium_room"),
        }

        self.log.info(f"音效设计: shot={shot_id}, foley={len(audio_hints['foley'])}, ambience={audio_hints['ambience']}")

        await self.report("done", {"shot_id": shot_id, "audio_hints": audio_hints})

        return {"status": "ok", "audio_hints": audio_hints}

    async def _design_scene(self, task: dict) -> dict:
        """为整个场景设计统一的环境音"""
        scene_id = task.get("scene_id", "")
        location = task.get("location", "")
        mood = task.get("mood", "neutral")
        style_guide = task.get("style_guide", {})

        ambience = self._get_or_create_ambience(scene_id, {"location_hint": location}, style_guide)

        # 匹配 BGM
        bgm_result = await self._select_bgm({"emotion": mood, "scene_id": scene_id})
        bgm_path = bgm_result.get("bgm_path", "")

        return {
            "status": "ok",
            "scene_id": scene_id,
            "ambience": ambience,
            "bgm_path": bgm_path,
        }

    async def _select_bgm(self, task: dict) -> dict:
        """根据情绪选择背景音乐"""
        emotion = task.get("emotion", "neutral")
        scene_id = task.get("scene_id", "")
        intensity = task.get("intensity", 0.5)

        # 检查缓存
        cache_key = f"{scene_id}:{emotion}"
        if cache_key in self._bgm_cache:
            return {"status": "ok", "bgm_path": self._bgm_cache[cache_key]}

        result = await self.call_tool("bgm_matcher", {
            "emotion": emotion,
            "scene_mood": task.get("mood", ""),
            "intensity": intensity,
        })

        data = result.data or {}
        bgm_path = data.get("bgm_path", "")

        if bgm_path and scene_id:
            self._bgm_cache[cache_key] = bgm_path

        self.log.info(f"BGM 选择: emotion={emotion} → {data.get('style', '')} ({data.get('bpm', 0)} bpm)")

        return {
            "status": "ok",
            "bgm_path": bgm_path,
            "bgm_info": {
                "bpm": data.get("bpm", 0),
                "key": data.get("key", ""),
                "style": data.get("style", ""),
                "loop": data.get("loop", False),
            },
        }

    async def _mix(self, task: dict) -> dict:
        """混音: 将对白、音效、BGM 混合为最终音轨"""
        shots = task.get("shots", [])
        dialog_clips = task.get("dialog_clips", {})
        sfx_designs = task.get("sfx_designs", {})
        bgm_path = task.get("bgm_path", "")
        output_path = task.get("output_path", "outputs/final_mix.wav")

        tracks = []

        # 对白轨
        for shot in shots:
            shot_id = shot.get("id", "")
            clip = dialog_clips.get(shot_id, {})
            if clip and clip.get("audio_path"):
                tracks.append({
                    "type": "dialog",
                    "path": clip["audio_path"],
                    "start_ms": int(shot.get("start_time", 0) * 1000),
                    "duration_ms": int(clip.get("duration", 0) * 1000),
                    "volume_db": -3.0,
                })

        # SFX 轨
        for shot in shots:
            shot_id = shot.get("id", "")
            design = sfx_designs.get(shot_id, {})
            for foley_path in design.get("foley", []):
                tracks.append({
                    "type": "sfx",
                    "path": foley_path,
                    "start_ms": int(shot.get("start_time", 0) * 1000),
                    "duration_ms": int(shot.get("duration", 3) * 1000),
                    "volume_db": -12.0,
                })
            ambience_path = design.get("ambience", "")
            if ambience_path:
                tracks.append({
                    "type": "ambience",
                    "path": ambience_path,
                    "start_ms": int(shot.get("start_time", 0) * 1000),
                    "duration_ms": int(shot.get("duration", 3) * 1000),
                    "volume_db": -18.0,
                })

        # BGM 轨
        if bgm_path:
            total_duration_ms = max(
                (int(s.get("start_time", 0) + s.get("duration", 3)) * 1000 for s in shots),
                default=0,
            )
            tracks.append({
                "type": "bgm",
                "path": bgm_path,
                "start_ms": 0,
                "duration_ms": total_duration_ms,
                "volume_db": -15.0,
            })

        # 调用混音工具
        mix_result = await self.call_tool("audio_mix", {
            "tracks": tracks,
            "output_path": output_path,
            "target_lufs": -23.0,
        })

        data = mix_result.data or {}
        final_path = data.get("audio_path", output_path)

        self.log.info(f"混音完成: {len(tracks)} 轨 → {final_path}")

        await self.report("done", {
            "output_path": final_path,
            "track_count": len(tracks),
            "duration": data.get("duration", 0),
        })

        return {
            "status": "ok",
            "audio_path": final_path,
            "tracks": len(tracks),
            "duration": data.get("duration", 0),
            "track_detail": {
                "dialog_tracks": sum(1 for t in tracks if t["type"] == "dialog"),
                "sfx_tracks": sum(1 for t in tracks if t["type"] == "sfx"),
                "ambience_tracks": sum(1 for t in tracks if t["type"] == "ambience"),
                "bgm_tracks": sum(1 for t in tracks if t["type"] == "bgm"),
            },
        }

    def _get_or_create_ambience(self, scene_id: str, shot: dict, style_guide: dict) -> dict:
        """获取或创建场景环境音"""
        if scene_id in self._scene_ambience:
            return self._scene_ambience[scene_id]

        location = ""
        if isinstance(shot.get("lighting"), dict):
            location = shot["lighting"].get("description", "")
        location = location or shot.get("location_hint", "")

        location_lower = location.lower()

        # 场景类型 → 环境音映射
        if "废墟" in location or "ruin" in location_lower:
            ambience = {"path": "sfx/wind_ambient.wav", "volume": 0.25, "reverb": "large_hall"}
        elif "室内" in location or "indoor" in location_lower:
            ambience = {"path": "sfx/silence_room_tone.wav", "volume": 0.15, "reverb": "small_room"}
        elif "森林" in location or "forest" in location_lower:
            ambience = {"path": "sfx/birds_ambient.wav", "volume": 0.2, "reverb": "outdoor"}
        elif "工厂" in location or "factory" in location_lower:
            ambience = {"path": "sfx/machinery_low_rumble.wav", "volume": 0.3, "reverb": "industrial_hall"}
        elif "雨" in location or "rain" in location_lower:
            ambience = {"path": "sfx/rain_light.wav", "volume": 0.22, "reverb": "outdoor_dampened"}
        elif "夜晚" in location or "night" in location_lower:
            ambience = {"path": "sfx/wind_ambient.wav", "volume": 0.15, "reverb": "outdoor"}
        else:
            visual_mood = style_guide.get("visual_mood", "")
            if "desolate" in visual_mood.lower() or "废墟" in str(shot):
                ambience = {"path": "sfx/wind_ambient.wav", "volume": 0.2, "reverb": "large_hall"}
            else:
                ambience = {"path": "sfx/silence_room_tone.wav", "volume": 0.15, "reverb": "medium_room"}

        self._scene_ambience[scene_id] = ambience
        return ambience
