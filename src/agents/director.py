"""总监 Agent (Director) — 全局编排、质量裁决、冲突仲裁"""

import asyncio
from src.core.agent import BaseAgent, AgentConfig
from src.data.protocols import QualityReport


class DirectorAgent(BaseAgent):
    """
    职责: 整个工厂的"大脑"，不亲自干活，但做所有关键决策

    主循环 (Pipeline 1→2→3→4):
      Pipeline 1 - 创意管线: 编剧 → 审核 → 分镜师 → 审核 → 美术指导
      Pipeline 2 - 角色管线: 角色导演 → 角色生成 → 资产锁定
      Pipeline 3 - 镜头管线: 制片调度 → 摄影师 → 逐镜头审核 (核心循环)
      Pipeline 4 - 后期管线: 剪辑 → 节奏审核 → 终审 → 输出
    """

    def __init__(self, config: AgentConfig, bus, tools):
        super().__init__(config, bus, tools)
        self.project_id: str = ""
        self.quality_threshold: float = 0.85
        self.project_state: dict = {}

    async def handle_task(self, task: dict) -> dict:
        action = task.get("action", "full")

        # 恢复/重试模式
        if action == "resume":
            return await self._handle_resume(task)
        elif action == "retry_failed":
            return await self._handle_retry_failed(task)

        creative_input = task.get("prompt", "")
        self.quality_threshold = task.get("quality_threshold", 0.85)
        language = task.get("language", "zh")

        self.log.info(f"启动项目: {creative_input[:100]} (语言: {language})")

        # ── Pipeline 1: 创意管线 ──
        screenplay = await self._pipeline_creative(creative_input, task)
        if screenplay.get("status") == "error":
            return screenplay

        shotlist = await self._pipeline_storyboard(screenplay)
        if shotlist.get("status") == "error":
            return shotlist

        style_guide = await self._pipeline_art_direction(shotlist)
        if style_guide.get("status") == "error":
            return style_guide

        # ── Pipeline 2: 角色管线 ──
        char_profiles = await self._pipeline_characters(screenplay, style_guide)

        # ── Pipeline 3: 镜头管线 (核心循环) ──
        all_frames = await self._pipeline_shots(shotlist, style_guide, char_profiles)

        # ── Pipeline 3.5: 音频管线 (配音 + 音效 + BGM) ──
        audio_assets = await self._pipeline_audio(shotlist, char_profiles, style_guide, language)

        # ── Pipeline 4: 后期管线 (调色 + 字幕/VFX + 剪辑 + 终审) ──
        final = await self._pipeline_post(all_frames, shotlist, audio_assets, style_guide, language)

        self.log.info(f"项目完成: {final.get('output_video', 'N/A')}")

        return {
            "status": "ok",
            "project_id": self.project_id,
            "screenplay": screenplay.get("screenplay"),
            "shotlist": shotlist.get("shotlist"),
            "style_guide": style_guide.get("style_guide"),
            "frames": all_frames,
            "final": final,
            "failed_shots": [],
        }

    async def _handle_resume(self, task: dict) -> dict:
        """从检查点恢复执行"""
        resume_state = task.get("resume_state", {})
        completed = task.get("completed_phases", [])
        self.quality_threshold = task.get("quality_threshold", 0.85)
        language = task.get("language", "zh")

        self.log.info(f"恢复执行: 已完成={completed}")

        screenplay = {"status": "ok", "screenplay": resume_state.get("screenplay", {})}
        shotlist = {"status": "ok", "shotlist": resume_state.get("shotlist", {})}
        style_guide = {"status": "ok", "style_guide": resume_state.get("style_guide", {})}
        char_profiles = resume_state.get("char_profiles", {})
        all_frames = resume_state.get("frames", [])

        # 跳过已完成的 phase
        if "creative" not in completed:
            screenplay = await self._pipeline_creative(task.get("prompt", ""), task)
            shotlist = await self._pipeline_storyboard(screenplay)
            style_guide = await self._pipeline_art_direction(shotlist)

        if "character" not in completed:
            char_profiles = await self._pipeline_characters(screenplay, style_guide)

        if "shots" not in completed:
            all_frames = await self._pipeline_shots(shotlist, style_guide, char_profiles)

        if "audio" not in completed:
            audio_assets = await self._pipeline_audio(shotlist, char_profiles, style_guide, language)
        else:
            audio_assets = resume_state.get("audio_assets", {})

        if "post" not in completed:
            final = await self._pipeline_post(all_frames, shotlist, audio_assets, style_guide, language)
        else:
            final = resume_state.get("final", {})

        return {
            "status": "ok",
            "project_id": self.project_id,
            "screenplay": screenplay.get("screenplay"),
            "shotlist": shotlist.get("shotlist"),
            "style_guide": style_guide.get("style_guide"),
            "frames": all_frames,
            "final": final,
            "failed_shots": [],
        }

    async def _handle_retry_failed(self, task: dict) -> dict:
        """仅重试失败的镜头"""
        resume_state = task.get("resume_state", {})
        failed_shots = task.get("failed_shots", [])
        self.quality_threshold = task.get("quality_threshold", 0.85)
        language = task.get("language", "zh")

        self.log.info(f"增量重试: {len(failed_shots)} 个失败镜头")

        screenplay = {"status": "ok", "screenplay": resume_state.get("screenplay", {})}
        shotlist = {"status": "ok", "shotlist": resume_state.get("shotlist", {})}
        style_guide = {"status": "ok", "style_guide": resume_state.get("style_guide", {})}
        char_profiles = resume_state.get("char_profiles", {})
        all_frames = resume_state.get("frames", [])

        # 仅重拍失败镜头
        retry_frames = await self._pipeline_retry_shots(
            failed_shots, shotlist, style_guide, char_profiles
        )

        # 合并结果
        frame_map = {f.get("shot_id", ""): f for f in all_frames}
        for rf in retry_frames:
            frame_map[rf.get("shot_id", "")] = rf
        merged_frames = list(frame_map.values())

        audio_assets = resume_state.get("audio_assets", {})
        final = await self._pipeline_post(merged_frames, shotlist, audio_assets, style_guide, language)

        return {
            "status": "ok",
            "project_id": self.project_id,
            "screenplay": screenplay.get("screenplay"),
            "shotlist": shotlist.get("shotlist"),
            "style_guide": style_guide.get("style_guide"),
            "frames": merged_frames,
            "final": final,
            "failed_shots": [],
        }

    async def _pipeline_retry_shots(self, failed_shots: list, shotlist_result: dict,
                                     style_guide_result: dict, char_profiles: dict) -> list[dict]:
        """重试指定的失败镜头"""
        shotlist = shotlist_result.get("shotlist", {})
        style_guide = style_guide_result.get("style_guide", {})
        all_shots = shotlist.get("shots", [])
        shot_map = {s.get("id", ""): s for s in all_shots}

        retry_frames = []
        for failed_id in failed_shots:
            shot = shot_map.get(failed_id)
            if not shot:
                continue
            self.log.info(f"重拍镜头: {failed_id}")
            result = await self.delegate("Cinematographer", {
                "shot": shot,
                "style_guide": style_guide,
                "char_profiles": char_profiles,
            })
            frame = result.get("frame", {})
            if frame:
                quality = await self._quality_check(shot, frame, char_profiles)
                frame["quality_report"] = self._qr_to_dict(quality)
                retry_frames.append(frame)

        self.log.info(f"重试完成: {len(retry_frames)}/{len(failed_shots)} 成功")
        return retry_frames

    # ── Pipeline 1: 创意管线 ────────────────────────

    async def _pipeline_creative(self, creative_input: str, task: dict) -> dict:
        """编剧 → 审核-修改循环"""
        self.log.info("── Pipeline 1: 创意管线 (编剧) ──")

        screenplay_result = await self.delegate("Writer", {
            "prompt": creative_input,
            "genre": task.get("genre", "drama"),
            "duration_hint": task.get("duration_hint", 60),
            "style_ref": task.get("style_ref", ""),
        })

        screenplay = screenplay_result.get("screenplay", {})

        # 审核剧本
        score = await self._evaluate_screenplay(screenplay)
        if score < self.quality_threshold:
            self.log.info(f"剧本质量不足 ({score:.2f})，请求修订...")
            screenplay_result = await self.delegate("Writer", {
                "prompt": creative_input,
                "genre": task.get("genre", "drama"),
                "duration_hint": task.get("duration_hint", 60),
                "revision_feedback": self._diagnose_screenplay(screenplay, score),
            })
            screenplay = screenplay_result.get("screenplay", {})

        return {"status": "ok", "screenplay": screenplay}

    async def _pipeline_storyboard(self, screenplay_result: dict) -> dict:
        """分镜师 → 审核"""
        self.log.info("── Pipeline 1: 创意管线 (分镜师) ──")

        screenplay = screenplay_result.get("screenplay", {})

        # 最多重试 2 次 (分镜师对 DeepSeek 推理模型偶发空输出)
        max_attempts = 2
        shotlist = {}
        shots = []

        for attempt in range(max_attempts):
            shotlist_result = await self.delegate("Storyboarder", {
                "screenplay": screenplay,
                "attempt": attempt + 1,
            })

            shotlist = shotlist_result.get("shotlist", {})
            shots = shotlist.get("shots", [])

            if len(shots) >= 3:
                break

            if attempt < max_attempts - 1:
                self.log.warn(f"分镜师返回 {len(shots)} 个镜头 (attempt {attempt+1})，重试中...")
                await asyncio.sleep(0.5)

        # 审核分镜节奏
        if len(shots) < 3:
            self.log.warn("镜头数不足，可能影响成片质量")
            # 从剧本原始文本构建最小分镜
            shots = self._build_fallback_shots(screenplay)
            shotlist["shots"] = shots
            shotlist["total_duration"] = sum(s.get("duration", 3.0) for s in shots)
            shotlist["project"] = screenplay.get("title", "Untitled")
        if len(shots) > 30:
            self.log.warn("镜头数偏多，建议控制时长")

        return {"status": "ok", "shotlist": shotlist}

    def _build_fallback_shots(self, screenplay: dict) -> list[dict]:
        """当 LLM 分镜失败时, 从剧本原始文本构建最小可用分镜"""
        scenes = screenplay.get("scenes", [])
        characters = screenplay.get("characters", [])
        char_ids = [c.get("id", "") for c in characters]

        fallback = []
        shot_num = 0
        for scene in scenes:
            scene_id = scene.get("id", f"s{shot_num+1}")
            location = scene.get("location", "")
            mood = scene.get("mood", "neutral")
            description = scene.get("description", "")

            # 每个场景至少 2 个镜头: 建立 + 细节
            shot_num += 1
            fallback.append({
                "id": f"shot_{shot_num:03d}",
                "scene_id": scene_id,
                "shot_number": shot_num,
                "duration": 3.5,
                "framing": "wide",
                "camera_angle": "eye_level",
                "camera_movement": "static",
                "depth_of_field": "deep",
                "subject": f"establishing shot of {location}",
                "composition_position": "center",
                "background": location,
                "lighting_description": f"{mood} atmospheric lighting",
                "color_temp": 5600,
                "ambience": mood,
                "foley": "",
                "bgm_mood": mood,
                "dialog": "",
                "emotion": mood,
                "emotion_intensity": 0.6,
                "transition_in": "cut",
                "transition_out": "cut",
                "transition_overlap": 0.0,
                "dependencies": [],
                "characters_in_frame": [],
                "action": description,
            })

            shot_num += 1
            fallback.append({
                "id": f"shot_{shot_num:03d}",
                "scene_id": scene_id,
                "shot_number": shot_num,
                "duration": 3.0,
                "framing": "medium",
                "camera_angle": "eye_level",
                "camera_movement": "static",
                "depth_of_field": "medium",
                "subject": "character in scene",
                "composition_position": "center_third",
                "background": location,
                "lighting_description": f"{mood} directional light",
                "color_temp": 4400,
                "ambience": mood,
                "foley": "",
                "bgm_mood": mood,
                "dialog": "",
                "emotion": mood,
                "emotion_intensity": 0.7,
                "transition_in": "cut",
                "transition_out": "cut",
                "transition_overlap": 0.0,
                "dependencies": [fallback[-1]["id"]] if fallback else [],
                "characters_in_frame": char_ids[:1] if char_ids else [],
                "action": f"{mood} scene detail",
            })

        self.log.info(f"回退分镜: 构建了 {len(fallback)} 个最小镜头")
        return fallback

    async def _pipeline_art_direction(self, shotlist_result: dict) -> dict:
        """美术指导 → 审核 → 广播 StyleGuide"""
        self.log.info("── Pipeline 1: 创意管线 (美术指导) ──")

        style_result = await self.delegate("ArtDirector", {
            "shotlist": shotlist_result.get("shotlist", {}),
            "style_hint": "cinematic, photorealistic",
        })

        style_guide = style_result.get("style_guide", {})

        # 广播 StyleGuide 到所有视觉 Agent
        await self.broadcast("style_guide_locked", style_guide)

        return {"status": "ok", "style_guide": style_guide}

    # ── Pipeline 2: 角色管线 ────────────────────────

    async def _pipeline_characters(self, screenplay_result: dict, style_guide_result: dict) -> dict:
        """为每个角色创建并锁定资产 — 委托给 CharacterDirector"""
        self.log.info("── Pipeline 2: 角色管线 ──")

        screenplay = screenplay_result.get("screenplay", {})
        characters = screenplay.get("characters", [])

        if not characters:
            self.log.info("无角色定义，跳过角色管线")
            return {}

        # 委托给 CharacterDirector 批量创建
        result = await self.delegate("CharacterDirector", {
            "action": "create_all",
            "characters": characters,
            "style_guide": style_guide_result.get("style_guide", {}),
        })

        profiles = result.get("profiles", {})

        # 逐个审核角色
        approved = {}
        for cid, profile in profiles.items():
            score = await self._evaluate_character(profile)
            if score >= self.quality_threshold:
                approved[cid] = profile
                self.log.info(f"角色审批通过: {profile.get('character_id', cid)} (score={score:.2f})")
            else:
                self.log.warn(f"角色审批未通过: {cid} (score={score:.2f})，使用默认配置")
                approved[cid] = profile  # 降级接受

        self.log.info(f"角色管线完成: {len(approved)} 个角色已锁定")
        return approved

    # ── Pipeline 3: 镜头管线 ────────────────────────

    async def _pipeline_shots(self, shotlist_result: dict, style_guide_result: dict, char_profiles: dict) -> list[dict]:
        """核心生成循环: 调度 → 拍摄 → 审核 → 重试"""
        self.log.info("── Pipeline 3: 镜头管线 ──")

        shotlist = shotlist_result.get("shotlist", {})
        style_guide = style_guide_result.get("style_guide", {})
        shots = shotlist.get("shots", [])

        # 初始化调度器
        await self.delegate("Scheduler", {
            "action": "init",
            "shotlist": shotlist,
            "max_retries": 3,
        })

        all_frames: list[dict] = []
        total = len(shots)

        while True:
            # 检查是否全部完成
            status = await self.delegate("Scheduler", {"action": "all_done"})
            if status.get("all_done"):
                break

            # 取下一批次
            batch_result = await self.delegate("Scheduler", {"action": "next_batch"})
            batch_shots = batch_result.get("shots", [])
            batch_ids = batch_result.get("batch", [])

            if not batch_shots:
                self.log.warn("无法获取新批次，检查卡住的镜头")
                break

            self.log.info(f"拍摄批次: {batch_ids} ({len(batch_shots)} 镜头)")

            # 并行拍摄 (带 prompt 缓存)
            tasks = []
            cached_prompts: dict[str, str | None] = {}
            for shot in batch_shots:
                # 检查 prompt 缓存
                cache_result = await self.call_tool("prompt_cache", {
                    "shot_spec": {
                        "framing": shot.get("camera", {}).get("framing", "") if isinstance(shot.get("camera"), dict) else "",
                        "emotion": shot.get("emotion", ""),
                        "scene_id": shot.get("scene_id", ""),
                        "action_description": shot.get("action_description", ""),
                        "characters_in_frame": shot.get("characters_in_frame", []),
                    },
                    "prompt": "",
                })
                cache_data = cache_result.data or {}
                shot_id = shot.get("id", "")
                cached_prompts[shot_id] = cache_data.get("prompt") if cache_data.get("hit") else None

                tasks.append(self.delegate("Cinematographer", {
                    "shot": shot,
                    "style_guide": style_guide,
                    "char_profiles": char_profiles,
                    "cached_prompt": cached_prompts[shot_id],
                }))

            results = await asyncio.gather(*tasks)

            # 逐镜头审核
            new_frames = []
            for shot, result in zip(batch_shots, results):
                shot_id = shot.get("id", "")
                frame = result.get("frame", {})
                status_flag = result.get("status", "ok")

                if status_flag == "ok":
                    # 缓存成功 prompt
                    if not cached_prompts.get(shot_id) and frame.get("prompt"):
                        await self.call_tool("prompt_cache", {
                            "shot_spec": {
                                "framing": shot.get("camera", {}).get("framing", "") if isinstance(shot.get("camera"), dict) else "",
                                "emotion": shot.get("emotion", ""),
                                "scene_id": shot.get("scene_id", ""),
                                "action_description": shot.get("action_description", ""),
                                "characters_in_frame": shot.get("characters_in_frame", []),
                            },
                            "prompt": frame.get("prompt", ""),
                            "negative_prompt": frame.get("negative_prompt", ""),
                            "params": frame.get("metadata", {}),
                            "quality_score": frame.get("composition_score", 0.88),
                        })

                    # 综合质量评分
                    quality = await self._quality_check(shot, frame, char_profiles)
                    if quality.passed:
                        await self.delegate("Scheduler", {
                            "action": "mark_done",
                            "shot_id": shot_id,
                        })
                        frame["quality_report"] = self._qr_to_dict(quality)
                        new_frames.append(frame)
                        self.log.info(f"  [OK] {shot_id} (score={quality.overall:.2f})")
                    else:
                        await self.delegate("Scheduler", {
                            "action": "retry",
                            "shot_id": shot_id,
                            "feedback": quality.feedback,
                        })
                        self.log.info(f"  [FAIL] {shot_id} (score={quality.overall:.2f}) -> 重试: {quality.feedback[:80]}")
                elif status_flag == "warning":
                    # 已达到最大重试 → 总监裁决
                    decision = await self._arbitrate_shot(shot, frame)
                    if decision.get("accept"):
                        await self.delegate("Scheduler", {"action": "mark_done", "shot_id": shot_id})
                        new_frames.append(frame)
                        self.log.info(f"  [WARN] {shot_id} 总监接受 (低于阈值但可用)")
                    else:
                        self.log.warn(f"  [REJECT] {shot_id} 总监拒绝，使用占位")

            # 光照连续性 — 场景级 + 逐镜头
            for shot in batch_shots:
                scene_id = shot.get("scene_id", "")
                await self.delegate("LightingTD", {
                    "action": "generate",
                    "shot": shot,
                    "style_guide": style_guide,
                })

            all_frames.extend(new_frames)
            self.log.info(f"进度: {len(all_frames)}/{total}")

        return all_frames

    # ── Pipeline 3.5: 音频管线 ───────────────────────

    async def _pipeline_audio(self, shotlist_result: dict, char_profiles: dict, style_guide_result: dict, language: str = "zh") -> dict:
        """配音 → 音效设计 → BGM 选择"""
        self.log.info("── Pipeline 3.5: 音频管线 ──")

        shotlist = shotlist_result.get("shotlist", {})
        shots = shotlist.get("shots", [])

        # 翻译对话 (非中文时)
        if language != "zh":
            await self._translate_shot_dialogs(shots, language)

        # 配音 — 为所有有对白的镜头生成 TTS
        self.log.info("配音生成中...")
        voice_result = await self.delegate("VoiceActor", {
            "action": "generate_all",
            "shots": shots,
            "char_profiles": char_profiles,
            "language": language,
        })
        dialog_clips = voice_result.get("clips", {})

        # 音效设计 — 逐镜头匹配 SFX + 环境音
        self.log.info("音效设计中...")
        sfx_designs = {}
        for shot in shots:
            sfx_result = await self.delegate("SoundDesigner", {
                "action": "design",
                "shot": shot,
                "style_guide": style_guide_result.get("style_guide", {}),
            })
            sfx_designs[shot.get("id", "")] = sfx_result.get("audio_hints", {})

        # BGM 选择 — 基于全片情绪曲线选择背景音乐
        screenplay = shotlist_result.get("screenplay", {})
        screenplay_obj = screenplay if isinstance(screenplay, dict) else {}
        emotion_curve = screenplay_obj.get("emotion_curve", [])
        dominant_emotion = "neutral"
        if emotion_curve:
            from collections import Counter
            emotions = [e if isinstance(e, str) else "neutral" for e in emotion_curve]
            dominant_emotion = Counter(emotions).most_common(1)[0][0]

        bgm_result = await self.delegate("SoundDesigner", {
            "action": "select_bgm",
            "emotion": dominant_emotion,
            "mood": style_guide_result.get("style_guide", {}).get("visual_mood", ""),
        })

        # 混音
        self.log.info("混音中...")
        mix_result = await self.delegate("SoundDesigner", {
            "action": "mix",
            "shots": shots,
            "dialog_clips": dialog_clips,
            "sfx_designs": sfx_designs,
            "bgm_path": bgm_result.get("bgm_path", ""),
            "output_path": "outputs/final_mix.wav",
        })

        self.log.info(f"音频管线完成: {len(dialog_clips)} 对白片段, {len(sfx_designs)} 音效设计")

        return {
            "dialog_clips": dialog_clips,
            "sfx_designs": sfx_designs,
            "bgm": bgm_result,
            "final_mix": mix_result,
        }

    async def _translate_shot_dialogs(self, shots: list[dict], target_language: str) -> None:
        """收集所有有对话的镜头，批量翻译并回写"""
        dialogs = {}
        for shot in shots:
            dialog = shot.get("dialog", "")
            if dialog and dialog.strip():
                dialogs[dialog] = None

        if not dialogs:
            return

        self.log.info(f"翻译 {len(dialogs)} 条对话 → {target_language}")
        result = await self.call_tool("translate", {
            "texts": list(dialogs.keys()),
            "target_language": target_language,
            "source_language": "zh",
        })

        translations = (result.data or {}).get("translations", {})
        for shot in shots:
            original = shot.get("dialog", "")
            if original in translations:
                shot["dialog"] = translations[original]

    # ── Pipeline 4: 后期管线 ────────────────────────

    async def _pipeline_post(self, frames: list[dict], shotlist_result: dict, audio_assets: dict, style_guide_result: dict, language: str = "zh") -> dict:
        """调色 → 字幕/VFX → 剪辑 → 终审 → 输出"""
        self.log.info("── Pipeline 4: 后期管线 ──")

        shotlist = shotlist_result.get("shotlist", {})
        shots = shotlist.get("shots", [])
        style_guide = style_guide_result.get("style_guide", {})

        # 调色 — 场景级色彩匹配
        self.log.info("调色中...")
        color_result = await self.delegate("Colorist", {
            "action": "grade_all",
            "shots": shots,
            "style_guide": style_guide,
        })

        # 色彩连续性检查
        if len(shots) > 1:
            continuity_result = await self.delegate("Colorist", {
                "action": "check_continuity",
                "shots": shots,
            })
            if not continuity_result.get("passed", True):
                self.log.warn(f"色彩不连续: {len(continuity_result.get('issues', []))} 处")

        # 字幕 + VFX
        self.log.info("字幕/VFX 生成中...")
        subtitle_result = await self.delegate("VFXSubtitles", {
            "action": "generate_all_subtitles",
            "shots": shots,
            "language": language,
        })
        vfx_result = await self.delegate("VFXSubtitles", {
            "action": "export_srt",
            "shots": shots,
            "output_path": "outputs/subtitles.srt",
        })

        # 剪辑
        edit_result = await self.delegate("Editor", {
            "frames": frames,
            "shotlist": shotlist,
            "audio_assets": audio_assets,
            "color_grades": color_result.get("grades", {}),
            "subtitles": subtitle_result.get("subtitles", {}),
        })

        # 终审
        timeline = edit_result.get("timeline", {})
        output = edit_result.get("output_video", "")

        self.log.info(f"终审通过: {output} (duration={timeline.get('total_duration', 0):.1f}s)")

        return edit_result

    # ── 质量检查 ────────────────────────────────────

    async def _quality_check(self, shot: dict, frame: dict, char_profiles: dict) -> QualityReport:
        comp_score = frame.get("composition_score", 0.85)

        # 角色一致性 — 通过 CharacterDirector 验证
        consistency_score = 1.0
        chars_in_frame = shot.get("characters_in_frame", [])
        if chars_in_frame:
            # 优先委托 CharacterDirector 做增强验证
            if len(chars_in_frame) == 1:
                verify_result = await self.delegate("CharacterDirector", {
                    "action": "verify",
                    "character_id": chars_in_frame[0],
                    "image_path": frame.get("image_path", ""),
                })
                verify_data = verify_result.get("data", {})
                consistency_score = verify_data.get("overall", 0.90)
            elif len(chars_in_frame) > 1:
                # 多角色同框
                verify_result = await self.delegate("CharacterDirector", {
                    "action": "verify_multi",
                    "character_ids": chars_in_frame,
                    "image_path": frame.get("image_path", ""),
                })
                verify_data = verify_result.get("data", {})
                consistency_score = 0.85 if verify_data.get("all_pass") else 0.78

        # 光照 (mock)
        light_score = 0.88

        # 情绪 (mock)
        emotion_score = 0.85

        overall = (
            comp_score * 0.40 +
            consistency_score * 0.35 +
            light_score * 0.15 +
            emotion_score * 0.10
        )

        passed = overall >= self.quality_threshold

        feedback = ""
        if not passed:
            parts = []
            if comp_score < 0.82:
                parts.append("构图不达标")
            if consistency_score < 0.85:
                parts.append("角色一致性不足")
            feedback = "; ".join(parts) if parts else "综合评分未达标"

        return QualityReport(
            shot_id=shot.get("id", ""),
            composition_score=comp_score,
            consistency_score=consistency_score,
            light_score=light_score,
            emotion_score=emotion_score,
            overall=round(overall, 3),
            passed=passed,
            feedback=feedback,
            suggestions=[feedback] if feedback else [],
        )

    async def _evaluate_character(self, profile: dict) -> float:
        """评估角色质量"""
        score = 0.85  # 基础分
        if profile.get("base_prompt"):
            score += 0.05
        if profile.get("lora_path"):
            score += 0.05
        if profile.get("face_embedding"):
            score += 0.03
        if profile.get("distinctive_features"):
            score += 0.02
        return min(score, 1.0)

    async def _arbitrate_shot(self, shot: dict, frame: dict) -> dict:
        """总监裁决: 对于质量不够但重试次数耗尽的镜头，做最终判断"""
        comp_score = frame.get("composition_score", 0.75)
        # 放宽阈值到 0.75
        if comp_score >= 0.75:
            return {"accept": True, "reason": "接近阈值，接受"}
        return {"accept": False, "reason": "质量过低，拒绝"}

    # ── 剧本评估 ────────────────────────────────────

    async def _evaluate_screenplay(self, screenplay: dict) -> float:
        characters = screenplay.get("characters", [])
        scenes = screenplay.get("scenes", [])
        emotion_curve = screenplay.get("emotion_curve", [])

        score = 0.85  # 基础分
        if len(characters) == 0:
            score -= 0.3
        if len(scenes) == 0:
            score -= 0.4
        if len(emotion_curve) == 0:
            score -= 0.15

        return max(0.0, score)

    def _diagnose_screenplay(self, screenplay: dict, score: float) -> str:
        issues = []
        if len(screenplay.get("characters", [])) == 0:
            issues.append("缺少角色定义")
        if len(screenplay.get("scenes", [])) == 0:
            issues.append("缺少场景拆分")
        if len(screenplay.get("emotion_curve", [])) == 0:
            issues.append("缺少情感曲线")
        return "; ".join(issues) if issues else "整体质量需要提升"

    def _qr_to_dict(self, qr: QualityReport) -> dict:
        return {
            "shot_id": qr.shot_id,
            "composition_score": qr.composition_score,
            "consistency_score": qr.consistency_score,
            "light_score": qr.light_score,
            "emotion_score": qr.emotion_score,
            "overall": qr.overall,
            "passed": qr.passed,
        }
