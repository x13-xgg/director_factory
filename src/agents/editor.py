"""剪辑师 Agent — 所有镜头 + ShotList → Timeline → 成品视频"""

from src.core.agent import BaseAgent, AgentConfig
from src.data.protocols import Timeline, TimelineClip, Transition, TransitionType


class EditorAgent(BaseAgent):
    """
    职责: 时间线装配
    不做生成，只做拼接和节奏控制
    """

    def __init__(self, config: AgentConfig, bus, tools):
        super().__init__(config, bus, tools)

    async def handle_task(self, task: dict) -> dict:
        frames = task.get("frames", [])       # [{shot_id, image_path, ...}]
        shotlist = task.get("shotlist", {})
        shots = shotlist.get("shots", [])
        project = shotlist.get("project", "Untitled")

        self.log.info(f"开始编辑: {len(frames)} 帧, {len(shots)} 镜头定义")

        # 构建 frame 索引
        frame_map = {f.get("shot_id", ""): f for f in frames}

        # 按 shotlist 顺序排列
        clips = []
        total_dur = 0.0

        for shot in shots:
            sid = shot.get("id", "")
            frame = frame_map.get(sid, {})
            duration = shot.get("duration", 3.0)
            transition_out = shot.get("transition_out", "cut")
            overlap = shot.get("transition_overlap", 0.0)

            # 根据情绪调整实际时长
            adjusted_duration = self._adjust_duration(duration, shot)

            clip = TimelineClip(
                shot_id=sid,
                video_path=frame.get("image_path", ""),
                in_point=0.0,
                out_point=adjusted_duration,
                transition=Transition(
                    type=TransitionType(transition_out) if transition_out else TransitionType.CUT,
                    overlap=overlap,
                ),
            )
            clips.append(clip)
            total_dur += adjusted_duration - overlap

        timeline = Timeline(
            project=project,
            clips=clips,
            total_duration=total_dur,
            fps=24,
        )

        # 调用 timeline_assemble 工具
        output_path = f"outputs/{project}_{self._timestamp()}.mp4"
        render_result = await self.call_tool("timeline_assemble", {
            "clips": [
                {
                    "video_path": c.video_path,
                    "in_point": c.in_point,
                    "out_point": c.out_point,
                    "duration": c.out_point - c.in_point,
                }
                for c in clips
            ],
            "transitions": [
                {"type": c.transition.type.value, "duration": c.transition.overlap}
                for c in clips
            ],
            "output_path": output_path,
            "fps": 24,
        })

        await self.report("done", {"duration": total_dur, "clips": len(clips), "output": output_path})

        return {
            "status": "ok",
            "timeline": {
                "project": timeline.project,
                "clips": [
                    {
                        "shot_id": c.shot_id,
                        "video_path": c.video_path,
                        "in_point": c.in_point,
                        "out_point": c.out_point,
                        "duration": c.out_point - c.in_point,
                        "transition": c.transition.type.value,
                    }
                    for c in clips
                ],
                "total_duration": timeline.total_duration,
                "fps": timeline.fps,
            },
            "output_video": render_result.data.get("video_path") if render_result.data else output_path,
        }

    def _adjust_duration(self, base_duration: float, shot: dict) -> float:
        """根据情绪和镜头类型微调时长"""
        emotion = shot.get("emotion", "neutral")
        intensity = shot.get("emotion_intensity", 0.5)
        framing = shot.get("framing", "medium")

        # 紧张情绪 → 缩短时长
        fast_emotions = ["tension", "fear", "anger", "surprise"]
        if emotion in fast_emotions:
            base_duration *= max(0.5, 1.0 - intensity * 0.4)

        # 特写 → 保持或微缩
        if framing in ["extreme_close_up", "close_up"]:
            base_duration *= 0.9

        # 广角 → 稍长
        if framing in ["extreme_wide", "wide"]:
            base_duration *= 1.1

        return round(max(1.0, base_duration), 1)

    def _timestamp(self) -> str:
        import time
        return time.strftime("%Y%m%d_%H%M%S")
