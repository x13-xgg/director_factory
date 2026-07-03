"""编剧 Agent — 用户创意 → 结构化 Screenplay"""

from src.core.agent import BaseAgent, AgentConfig
from src.data.protocols import Screenplay, Character, Scene, Emotion


class WriterAgent(BaseAgent):
    """
    职责: 将用户的一句话/梗概扩展为结构化剧本
    关键: 用"镜头语言"而非"文学语言"写作
    """

    def __init__(self, config: AgentConfig, bus, tools):
        super().__init__(config, bus, tools)

    async def handle_task(self, task: dict) -> dict:
        prompt = task.get("prompt", "")
        genre = task.get("genre", "drama")
        duration_hint = task.get("duration_hint", 60)
        style_ref = task.get("style_ref", "")

        self.log.info(f"开始创作剧本: {prompt[:100]}")

        result = await self.call_tool("text_gen", {
            "system_prompt": self._build_system_prompt(),
            "user_prompt": self._build_user_prompt(prompt, genre, duration_hint, style_ref),
            "output_schema": ScreenplaySchema,
            "temperature": 0.8,
            "max_tokens": 4096,
        })

        # 解析结果
        data = result.data
        if isinstance(data, dict):
            screenplay = Screenplay(
                title=data.get("title", "Untitled"),
                genre=genre,
                target_duration=duration_hint,
                characters=[Character(**c) for c in data.get("characters", [])],
                scenes=[Scene(**s) for s in data.get("scenes", [])],
                emotion_curve=data.get("emotion_curve", []),
                raw_text=data.get("raw_text", ""),
            )
        else:
            screenplay = Screenplay(title="Untitled", genre=genre, target_duration=duration_hint)

        await self.report("done", {"title": screenplay.title, "characters": len(screenplay.characters), "scenes": len(screenplay.scenes)})

        return {
            "status": "ok",
            "screenplay": self._screenplay_to_dict(screenplay),
        }

    def _build_system_prompt(self) -> str:
        return """你是一位专业的影视编剧。你的任务是把一个创意概念扩展为一个结构化的短剧本。

写作要求:
1. 用"镜头语言"写作，每个段落 = 一个可拍摄的镜头
2. 标注景别（广角/中景/特写/极特写）、机位、运镜方式
3. 每个场景标注地点、情绪、时间
4. 为每个角色标注外貌特征和声音风格
5. 设计情感曲线：标注关键情节点的情绪强度 (0-1)

输出格式严格遵循指定的 JSON schema。"""

    def _build_user_prompt(self, prompt: str, genre: str, duration: float, style: str) -> str:
        parts = [
            f"创意: {prompt}",
            f"类型: {genre}",
            f"目标时长: {duration} 秒",
        ]
        if style:
            parts.append(f"风格参考: {style}")
        parts.append("请生成7-15个镜头的短剧本，确保每个镜头都包含完整的景别、机位、运镜和情绪标注。")
        return "\n".join(parts)

    def _screenplay_to_dict(self, sp: Screenplay) -> dict:
        return {
            "title": sp.title,
            "genre": sp.genre,
            "target_duration": sp.target_duration,
            "characters": [{"id": c.id, "name": c.name, "description": c.description, "voice_style": c.voice_style} for c in sp.characters],
            "scenes": [{"id": s.id, "location": s.location, "mood": s.mood, "time_of_day": s.time_of_day, "shots": s.shots, "description": s.description} for s in sp.scenes],
            "emotion_curve": sp.emotion_curve,
            "raw_text": sp.raw_text,
            "version": sp.version,
        }


# ── JSON Schema ─────────────────────────────────────

ScreenplaySchema = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "剧本标题"},
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string", "description": "外貌特征"},
                    "voice_style": {"type": "string", "description": "声音风格"},
                },
                "required": ["id", "name", "description", "voice_style"],
            },
        },
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "location": {"type": "string"},
                    "mood": {"type": "string"},
                    "time_of_day": {"type": "string", "enum": ["dawn", "day", "dusk", "night"]},
                    "shots": {"type": "array", "items": {"type": "string"}},
                    "description": {"type": "string"},
                },
                "required": ["id", "location", "mood", "shots"],
            },
        },
        "emotion_curve": {"type": "array", "items": {"type": "number"}, "description": "情绪强度随时间变化曲线"},
        "raw_text": {"type": "string", "description": "带镜头标注的剧本正文"},
    },
    "required": ["title", "characters", "scenes", "emotion_curve", "raw_text"],
}
