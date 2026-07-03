"""调色师 Agent — 场景级色彩匹配 + LUT 应用 + 镜头间色彩连续性"""

from __future__ import annotations

from src.core.agent import BaseAgent, AgentConfig


class ColoristAgent(BaseAgent):
    """
    职责:
      1. 根据 StyleGuide 中的 VisualSpec 为每个场景调色
      2. 应用 LUT (Look-Up Table) 预设
      3. 检查相邻镜头色彩连续性 (色温、色调、对比度)
      4. 输出逐镜头调色参数，供后期合成使用
    """

    def __init__(self, config: AgentConfig, bus, tools):
        super().__init__(config, bus, tools)
        self._scene_grades: dict[str, dict] = {}
        self._shot_grades: dict[str, dict] = {}

    async def handle_task(self, task: dict) -> dict:
        action = task.get("action", "grade_shot")

        if action == "grade_shot":
            return await self._grade_shot(task)
        elif action == "grade_scene":
            return await self._grade_scene(task)
        elif action == "grade_all":
            return await self._grade_all(task)
        elif action == "check_continuity":
            return await self._check_color_continuity(task)
        else:
            return {"status": "error", "error": f"Unknown action: {action}"}

    async def _grade_shot(self, task: dict) -> dict:
        """为单个镜头调色"""
        shot = task.get("shot", {})
        shot_id = shot.get("id", "")
        scene_id = shot.get("scene_id", "")
        style_guide = task.get("style_guide", {})

        # 获取场景级调色基底
        scene_grade = self._get_or_create_scene_grade(scene_id, shot, style_guide)

        # 在场景基础上做镜头级微调
        shot_grade = self._adapt_to_shot(scene_grade, shot)

        # 调用调色工具
        result = await self.call_tool("color_grade", {
            "image_path": f"outputs/frames/{shot_id}.png",
            "palette_dominant": shot_grade.get("palette_dominant", ""),
            "palette_accent": shot_grade.get("palette_accent", ""),
            "mood_descriptor": shot_grade.get("mood", ""),
            "color_temp_k": shot_grade.get("color_temp_k", 5600),
            "scene_id": scene_id,
        })

        data = result.data or {}
        self._shot_grades[shot_id] = data

        self.log.info(f"调色: shot={shot_id}, lut={data.get('lut_applied', '')}, temp={data.get('grade_params', {}).get('temperature', 0)}")

        await self.report("done", {"shot_id": shot_id, "grade_params": data.get("grade_params", {})})

        return {
            "status": "ok",
            "shot_id": shot_id,
            "grade_params": data.get("grade_params", {}),
            "lut_applied": data.get("lut_applied", ""),
        }

    async def _grade_scene(self, task: dict) -> dict:
        """为整个场景建立调色基底"""
        scene_id = task.get("scene_id", "")
        style_guide = task.get("style_guide", {})
        visual_specs = style_guide.get("visual_specs", {})
        spec = visual_specs.get(scene_id, {})

        grade = {
            "scene_id": scene_id,
            "palette_dominant": spec.get("palette_dominant", ""),
            "palette_accent": spec.get("palette_accent", ""),
            "mood": spec.get("mood_descriptor", ""),
            "color_temp_k": 5600,
        }

        self._scene_grades[scene_id] = grade

        return {"status": "ok", "scene_grade": grade}

    async def _grade_all(self, task: dict) -> dict:
        """批量为所有镜头调色"""
        shots = task.get("shots", [])
        style_guide = task.get("style_guide", {})

        grades = {}
        for shot in shots:
            shot_id = shot.get("id", "")
            result = await self._grade_shot({
                "shot": shot,
                "style_guide": style_guide,
            })
            grades[shot_id] = result

        self.log.info(f"批量调色完成: {len(grades)} 个镜头")

        return {
            "status": "ok",
            "grades": grades,
            "count": len(grades),
        }

    async def _check_color_continuity(self, task: dict) -> dict:
        """检查相邻镜头色彩连续性"""
        shots = task.get("shots", [])
        threshold = task.get("threshold", 0.15)

        issues = []
        for i in range(1, len(shots)):
            prev_id = shots[i - 1].get("id", "")
            curr_id = shots[i].get("id", "")
            prev_grade = self._shot_grades.get(prev_id, {})
            curr_grade = self._shot_grades.get(curr_id, {})

            prev_params = prev_grade.get("grade_params", {})
            curr_params = curr_grade.get("grade_params", {})

            temp_diff = abs(prev_params.get("temperature", 0) - curr_params.get("temperature", 0))
            sat_diff = abs(prev_params.get("saturation", 1.0) - curr_params.get("saturation", 1.0))
            cont_diff = abs(prev_params.get("contrast", 1.0) - curr_params.get("contrast", 1.0))

            if temp_diff > threshold or sat_diff > threshold or cont_diff > threshold:
                issues.append({
                    "prev_shot": prev_id,
                    "current_shot": curr_id,
                    "temp_drift": round(temp_diff, 3),
                    "saturation_drift": round(sat_diff, 3),
                    "contrast_drift": round(cont_diff, 3),
                })
                self.log.info(f"色彩不连续: {prev_id}→{curr_id} (temp={temp_diff:.3f}, sat={sat_diff:.3f})")

        passed = len(issues) == 0

        await self.report("done" if passed else "continuity_issue", {
            "passed": passed,
            "issues": issues,
        })

        return {
            "status": "ok",
            "passed": passed,
            "issues": issues,
            "checked_pairs": max(0, len(shots) - 1),
        }

    def _get_or_create_scene_grade(self, scene_id: str, shot: dict, style_guide: dict) -> dict:
        """获取或创建场景调色基底"""
        if scene_id in self._scene_grades:
            return self._scene_grades[scene_id]

        visual_specs = style_guide.get("visual_specs", {})
        spec = visual_specs.get(scene_id, {})

        # 从视觉规范中提取色彩信息
        grade = {
            "scene_id": scene_id,
            "palette_dominant": spec.get("palette_dominant", ""),
            "palette_accent": spec.get("palette_accent", ""),
            "mood": spec.get("mood_descriptor", style_guide.get("visual_mood", "")),
            "color_temp_k": shot.get("lighting", {}).get("color_temp_k", 5600) if isinstance(shot.get("lighting"), dict) else 5600,
        }

        self._scene_grades[scene_id] = grade
        return grade

    def _adapt_to_shot(self, scene_grade: dict, shot: dict) -> dict:
        """在场景调色基底上做镜头级微调"""
        import copy
        grade = copy.deepcopy(scene_grade)

        emotion = shot.get("emotion", "neutral")

        # 情绪微调色温
        emotion_temp_shift = {
            "anger": 200, "fear": -300, "sadness": -150, "joy": 100,
            "hope": 50, "loneliness": -250, "tension": -100, "surprise": -50,
            "wistful": 100, "serene": 50, "neutral": 0,
        }
        grade["color_temp_k"] = grade.get("color_temp_k", 5600) + emotion_temp_shift.get(emotion, 0)

        # 运镜微调
        movement = shot.get("camera_movement", "static")
        if hasattr(movement, 'value'):
            movement = movement.value
        if movement == "handheld":
            grade["color_temp_k"] -= 100

        return grade
