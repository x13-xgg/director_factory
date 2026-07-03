"""字幕/VFX Agent — 字幕生成 + 视觉特效叠加"""

from __future__ import annotations

from src.core.agent import BaseAgent, AgentConfig


class VFXSubtitlesAgent(BaseAgent):
    """
    职责:
      1. 从对白文本生成 SRT 格式字幕
      2. 根据 scene mood 自动推荐 VFX 效果
      3. 字幕样式管理 (字体、位置、大小)
      4. VFX 参数生成 (粒子、光晕、抖动等)
    """

    def __init__(self, config: AgentConfig, bus, tools):
        super().__init__(config, bus, tools)
        self._subtitle_cache: dict[str, dict] = {}
        self._vfx_cache: dict[str, dict] = {}
        self._subtitle_style: dict = {
            "font": "Noto Sans SC",
            "font_size": 36,
            "color": "#FFFFFF",
            "outline_color": "#000000",
            "outline_width": 2,
            "position": "bottom",
            "alignment": "center",
        }

    async def handle_task(self, task: dict) -> dict:
        action = task.get("action", "generate_subtitles")

        if action == "generate_subtitles":
            return await self._generate_subtitles(task)
        elif action == "generate_all_subtitles":
            return await self._generate_all_subtitles(task)
        elif action == "apply_vfx":
            return await self._apply_vfx(task)
        elif action == "process_shot":
            return await self._process_shot(task)
        elif action == "set_style":
            return await self._set_style(task)
        elif action == "export_srt":
            return await self._export_srt(task)
        else:
            return {"status": "error", "error": f"Unknown action: {action}"}

    async def _generate_subtitles(self, task: dict) -> dict:
        """为单个镜头生成字幕"""
        shot = task.get("shot", {})
        shot_id = shot.get("id", "")
        dialog = shot.get("dialog", "")
        duration = shot.get("duration", 3.0)
        scene_mood = shot.get("emotion", "")
        language = task.get("language", "zh")

        if not dialog:
            return {"status": "ok", "shot_id": shot_id, "subtitle": None, "note": "no dialog"}

        result = await self.call_tool("vfx_subtitle", {
            "dialog": dialog,
            "shot_id": shot_id,
            "duration": duration,
            "vfx_type": "none",
            "scene_mood": scene_mood,
            "language": language,
        })

        data = result.data or {}
        subtitle = {
            "shot_id": shot_id,
            "srt_content": data.get("srt_content", ""),
            "subtitle_count": data.get("subtitle_count", 0),
            "style": dict(self._subtitle_style),
        }

        self._subtitle_cache[shot_id] = subtitle

        self.log.info(f"字幕生成: shot={shot_id}, lines={subtitle['subtitle_count']}")

        return {"status": "ok", "shot_id": shot_id, "subtitle": subtitle}

    async def _generate_all_subtitles(self, task: dict) -> dict:
        """批量为所有镜头生成字幕"""
        shots = task.get("shots", [])
        language = task.get("language", "zh")

        subtitles = {}
        for shot in shots:
            result = await self._generate_subtitles({"shot": shot, "language": language})
            if result.get("subtitle"):
                subtitles[shot.get("id", "")] = result["subtitle"]

        self.log.info(f"批量字幕生成完成: {len(subtitles)} 个镜头")

        return {
            "status": "ok",
            "subtitles": subtitles,
            "count": len(subtitles),
        }

    async def _apply_vfx(self, task: dict) -> dict:
        """为镜头应用 VFX"""
        shot = task.get("shot", {})
        shot_id = shot.get("id", "")
        vfx_type = task.get("vfx_type", "")
        scene_mood = task.get("scene_mood", shot.get("emotion", ""))

        # 如果未指定 vfx_type，从 mood 自动推导
        if not vfx_type:
            result = await self.call_tool("vfx_subtitle", {
                "dialog": "",
                "shot_id": shot_id,
                "duration": shot.get("duration", 3.0),
                "vfx_type": "none",
                "scene_mood": scene_mood,
            })
            data = result.data or {}
            vfx_type = data.get("auto_vfx_suggestion", "")

        vfx_config = self._build_vfx_config(vfx_type, shot)

        result = await self.call_tool("vfx_subtitle", {
            "dialog": "",
            "shot_id": shot_id,
            "duration": shot.get("duration", 3.0),
            "vfx_type": vfx_type,
            "scene_mood": scene_mood,
        })

        data = result.data or {}
        vfx_result = {
            "shot_id": shot_id,
            "vfx_type": vfx_type,
            "params": data.get("vfx_params", {}),
            "config": vfx_config,
        }

        self._vfx_cache[shot_id] = vfx_result

        self.log.info(f"VFX 应用: shot={shot_id}, type={vfx_type}")

        return {"status": "ok", "shot_id": shot_id, "vfx": vfx_result}

    async def _process_shot(self, task: dict) -> dict:
        """一站式处理: 字幕 + VFX"""
        shot = task.get("shot", {})

        sub_result = await self._generate_subtitles({"shot": shot})
        vfx_result = await self._apply_vfx({"shot": shot})

        return {
            "status": "ok",
            "shot_id": shot.get("id", ""),
            "subtitle": sub_result.get("subtitle"),
            "vfx": vfx_result.get("vfx"),
        }

    async def _set_style(self, task: dict) -> dict:
        """设置字幕样式"""
        style_updates = task.get("style", {})
        self._subtitle_style.update(style_updates)
        return {"status": "ok", "style": dict(self._subtitle_style)}

    async def _export_srt(self, task: dict) -> dict:
        """导出完整 SRT 文件"""
        shots = task.get("shots", [])
        output_path = task.get("output_path", "outputs/subtitles.srt")

        full_srt = ""
        subtitle_index = 0
        cumulative_time = 0.0

        for shot in shots:
            shot_id = shot.get("id", "")
            duration = shot.get("duration", 3.0)
            cached = self._subtitle_cache.get(shot_id, {})

            if cached.get("srt_content"):
                # 重新调整时间戳
                srt_lines = cached["srt_content"].strip().split("\n\n")
                for block in srt_lines:
                    if not block.strip():
                        continue
                    parts = block.split("\n")
                    if len(parts) >= 3:
                        subtitle_index += 1
                        # 保留原始时间偏移
                        full_srt += f"{subtitle_index}\n"
                        full_srt += f"{parts[1]}\n"
                        full_srt += f"{parts[2]}\n\n"
                cumulative_time += duration
            else:
                cumulative_time += duration

        # 写入文件
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(full_srt, encoding="utf-8")

        self.log.info(f"SRT 导出: {output_path} ({subtitle_index} 条字幕)")

        return {
            "status": "ok",
            "srt_path": output_path,
            "subtitle_count": subtitle_index,
        }

    def _build_vfx_config(self, vfx_type: str, shot: dict) -> dict:
        """构建 VFX 配置参数"""
        configs = {
            "film_grain": {"intensity": 0.15, "size": 1.5, "monochrome": True},
            "vignette_dark": {"intensity": 0.4, "feather": 0.6, "roundness": 0.0},
            "light_leak": {"intensity": 0.3, "color": "#FFD700", "direction": "top_right"},
            "dust_particles": {"count": 50, "size": 2.0, "speed": 0.3, "opacity": 0.4},
            "subtle_blur": {"radius": 1.5, "type": "gaussian", "edge_preserve": True},
            "chromatic_aberration": {"shift": 2.0, "direction": "radial"},
            "lens_flare": {"intensity": 0.25, "position": [0.8, 0.2], "color": "#FFFFFF"},
            "scan_line": {"spacing": 2, "opacity": 0.08, "animation": "scroll"},
            "glitch": {"intensity": 0.3, "frequency": 0.1, "block_size": 16},
            "none": {},
        }
        return configs.get(vfx_type, {})
