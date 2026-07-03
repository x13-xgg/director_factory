"""Pipeline Runner — 组装所有 Agent 和 Tool，运行完整管线"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from src.core.agent import AgentConfig
from src.core.message_bus import MessageBus
from src.core.logging import get_logger, tracer
from src.tools.base import ToolRegistry
from src.tools.text_gen import TextGenTool
from src.tools.image_gen import ComfyUIImageGenTool
from src.tools.scorers import (
    CompositionScorerTool,
    FaceConsistencyCheckerTool,
    LightContinuityCheckerTool,
    EmotionAlignmentCheckerTool,
    RhythmScorerTool,
    QualityAggregatorTool,
    BenchmarkTool,
)
from src.tools.render import TimelineAssembleTool, AudioMixTool
from src.tools.character_tools import (
    LoRATrainerTool,
    EmbedExtractorTool,
    CharacterConsistencyCheckerTool,
    MultiCharacterCompositionTool,
)
from src.tools.audio_video import (
    TTSTool,
    SFXMatcherTool,
    BGMMatcherTool,
    ColorGradeTool,
    VFXSubtitleTool,
)
from src.tools.translator import TranslationTool
from src.tools.performance import (
    PromptCacheTool,
    GPUSchedulerTool,
    CheckpointTool,
)
from src.tools.asset_db import asset_db

from src.agents.director import DirectorAgent
from src.agents.writer import WriterAgent
from src.agents.storyboarder import StoryboarderAgent
from src.agents.art_director import ArtDirectorAgent
from src.agents.character_director import CharacterDirectorAgent
from src.agents.cinematographer import CinematographerAgent
from src.agents.scheduler import SchedulerAgent
from src.agents.editor import EditorAgent
from src.agents.lighting_td import LightingTDAgent
from src.agents.voice_actor import VoiceActorAgent
from src.agents.sound_designer import SoundDesignerAgent
from src.agents.colorist import ColoristAgent
from src.agents.vfx_subtitles import VFXSubtitlesAgent


class PipelineRunner:
    """
    管线执行器 — 负责:
      1. 初始化 MessageBus
      2. 注册所有 Tool
      3. 实例化所有 Agent
      4. 执行完整管线
    """

    def __init__(self, output_dir: str = "outputs"):
        self.bus = MessageBus()
        self.tools = ToolRegistry()
        self.agents: dict[str, Any] = {}
        self.log = get_logger("PipelineRunner")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def setup(self):
        """初始化所有工具和 Agent"""
        self.log.info("=" * 60)
        self.log.info("初始化导演工厂 Pipeline")
        self.log.info("=" * 60)

        # ── 从 PostgreSQL 加载持久化资产 (如果已配置) ──
        await asset_db.pg_load_all()

        # ── 注册工具 ──
        self._register_tools()

        # ── 注册 Agent ──
        self._register_agents()

        self.log.info(f"已注册 {len(self.tools.list_tools())} 个工具")
        self.log.info(f"已注册 {len(self.agents)} 个 Agent")

    def _register_tools(self):
        """注册所有工具到 ToolRegistry"""
        tools = [
            # 生成类
            TextGenTool(),
            ComfyUIImageGenTool(),
            # 审核类
            CompositionScorerTool(),
            FaceConsistencyCheckerTool(),
            LightContinuityCheckerTool(),
            EmotionAlignmentCheckerTool(),
            RhythmScorerTool(),
            QualityAggregatorTool(),
            BenchmarkTool(),
            # 角色类
            LoRATrainerTool(),
            EmbedExtractorTool(),
            CharacterConsistencyCheckerTool(),
            MultiCharacterCompositionTool(),
            # 渲染类
            TimelineAssembleTool(),
            AudioMixTool(),
            # 音频/视频后期类
            TTSTool(),
            SFXMatcherTool(),
            BGMMatcherTool(),
            ColorGradeTool(),
            VFXSubtitleTool(),
            # 翻译
            TranslationTool(),
            # 性能/调度类
            PromptCacheTool(),
            GPUSchedulerTool(),
            CheckpointTool(),
        ]
        for t in tools:
            self.tools.register(t)

    def _register_agents(self):
        """实例化所有 Agent"""
        agent_classes = {
            "Director": DirectorAgent,
            "Writer": WriterAgent,
            "Storyboarder": StoryboarderAgent,
            "ArtDirector": ArtDirectorAgent,
            "CharacterDirector": CharacterDirectorAgent,
            "Cinematographer": CinematographerAgent,
            "Scheduler": SchedulerAgent,
            "Editor": EditorAgent,
            "LightingTD": LightingTDAgent,
            "VoiceActor": VoiceActorAgent,
            "SoundDesigner": SoundDesignerAgent,
            "Colorist": ColoristAgent,
            "VFXSubtitles": VFXSubtitlesAgent,
        }

        for name, cls in agent_classes.items():
            config = AgentConfig(
                name=name,
                role=name.lower(),
                model="claude-sonnet-4-6",
                max_retries=3,
                temperature=0.7 if name != "Storyboarder" else 0.5,
            )
            agent = cls(config, self.bus, self.tools)
            self.agents[name] = agent

    async def run(self, creative_input: str, **kwargs) -> dict:
        """
        执行完整管线

        参数:
          creative_input: 用户创意 (一句话/一段文本)
          genre: 类型 (默认 "drama")
          duration_hint: 目标时长秒数 (默认 60)
          style_ref: 风格参考 (可选)
          quality_threshold: 质量阈值 (默认 0.85)
        """
        await self.setup()

        # 启动所有 Agent 的消息循环 (后台)
        agent_tasks = []
        for name, agent in self.agents.items():
            task = asyncio.create_task(agent.run_loop())
            agent_tasks.append(task)

        # 给 Agent 一点时间完成 start()
        await asyncio.sleep(0.1)

        # 委托总监执行
        director = self.agents["Director"]
        task_payload = {
            "prompt": creative_input,
            "genre": kwargs.get("genre", "drama"),
            "duration_hint": kwargs.get("duration_hint", 60),
            "style_ref": kwargs.get("style_ref", ""),
            "quality_threshold": kwargs.get("quality_threshold", 0.85),
            "language": kwargs.get("language", "zh"),
        }

        self.log.info(f"运行管线: {creative_input[:80]}...")
        result = await self.bus.request("system", "Director", "task", task_payload, timeout=600.0)

        if result is None:
            # fallback: 直接调用
            result = await director.handle_task(task_payload)

        # 关闭所有 Agent
        for name in self.agents:
            await self.bus.send("system", name, "shutdown", {})

        # 等待关闭
        await asyncio.sleep(0.2)
        for t in agent_tasks:
            t.cancel()

        # 关闭 PG 连接池
        await asset_db.close()

        # 保存追踪
        tracer.flush(self.output_dir / "trace.json")

        # 保存结果摘要
        summary = self._build_summary(result or {})
        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        self.log.info(f"结果已保存: {summary_path}")

        return summary

    def _build_summary(self, result: dict) -> dict:
        """构建结果摘要"""
        screenplay = result.get("screenplay", {})
        shotlist = result.get("shotlist", {})
        frames = result.get("frames", [])
        final = result.get("final", {})

        return {
            "title": screenplay.get("title", "Untitled"),
            "status": result.get("status", "unknown"),
            "stats": {
                "characters": len(screenplay.get("characters", [])),
                "scenes": len(screenplay.get("scenes", [])),
                "shots_planned": len(shotlist.get("shots", [])),
                "shots_completed": len(frames),
                "total_duration": shotlist.get("total_duration", 0),
            },
            "output_video": final.get("output_video", ""),
            "frames": frames,
            "trace_file": str(self.output_dir / "trace.json"),
        }

    # ── 检查点/恢复 ──────────────────────────────────

    async def resume(self, project_id: str, **kwargs) -> dict:
        """从检查点恢复管线执行"""
        await self.setup()
        self.log.info(f"尝试从检查点恢复: {project_id}")

        # 加载检查点
        cp_result = await self.tools.call("checkpoint", {
            "action": "load",
            "project_id": project_id,
        })
        cp_data = cp_result.data or {}
        if not cp_data.get("found"):
            self.log.info(f"检查点不存在，执行完整管线: {project_id}")
            return await self.run(kwargs.get("creative_input", project_id), **kwargs)

        state = cp_data.get("state", {})
        completed = cp_data.get("completed_phases", [])
        self.log.info(f"恢复: 已完成阶段={completed}, 已完成镜头={cp_data.get('completed_shot_count', 0)}")

        # 启动 Agent
        agent_tasks = []
        for name, agent in self.agents.items():
            task = asyncio.create_task(agent.run_loop())
            agent_tasks.append(task)
        await asyncio.sleep(0.1)

        director = self.agents["Director"]

        # 构建恢复任务
        task_payload = {
            "action": "resume",
            "project_id": project_id,
            "resume_state": state,
            "completed_phases": completed,
            "prompt": kwargs.get("prompt", ""),
            "genre": kwargs.get("genre", "drama"),
            "duration_hint": kwargs.get("duration_hint", 60),
            "style_ref": kwargs.get("style_ref", ""),
            "quality_threshold": kwargs.get("quality_threshold", 0.85),
            "language": kwargs.get("language", "zh"),
        }

        result = await director.handle_task(task_payload)

        # 保存检查点
        self._save_checkpoint_inline(project_id, result)

        await self._shutdown_agents(agent_tasks)
        return self._build_summary(result)

    async def retry_failed(self, project_id: str) -> dict:
        """仅重试失败的镜头 (增量模式)"""
        await self.setup()
        self.log.info(f"增量重试模式: {project_id}")

        cp_result = await self.tools.call("checkpoint", {
            "action": "load",
            "project_id": project_id,
        })
        cp_data = cp_result.data or {}
        if not cp_data.get("found"):
            return {"status": "error", "error": f"Checkpoint not found: {project_id}"}

        state = cp_data.get("state", {})
        failed_shots = state.get("failed_shots", [])
        if not failed_shots:
            self.log.info("无失败镜头，无需重试")
            return self._build_summary(state)

        self.log.info(f"增量重试: {len(failed_shots)} 个失败镜头")

        agent_tasks = []
        for name, agent in self.agents.items():
            task = asyncio.create_task(agent.run_loop())
            agent_tasks.append(task)
        await asyncio.sleep(0.1)

        director = self.agents["Director"]
        result = await director.handle_task({
            "action": "retry_failed",
            "project_id": project_id,
            "resume_state": state,
            "failed_shots": failed_shots,
        })

        self._save_checkpoint_inline(project_id, result)
        await self._shutdown_agents(agent_tasks)
        return self._build_summary(result)

    async def _shutdown_agents(self, agent_tasks: list):
        for name in self.agents:
            await self.bus.send("system", name, "shutdown", {})
        await asyncio.sleep(0.2)
        for t in agent_tasks:
            t.cancel()
        await asset_db.close()

    def _save_checkpoint_inline(self, project_id: str, state: dict):
        import time
        cp_path = self.output_dir / "checkpoints" / f"{project_id}.json"
        cp_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "project_id": project_id,
            "saved_at": time.time(),
            "saved_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "state": state,
        }
        cp_path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
        self.log.info(f"检查点已保存: {cp_path}")

    async def run_interactive(self):
        """交互模式: 接受用户输入并执行"""
        await self.setup()
        print("\n=== 全自动导演工厂 - 交互模式 ===")
        print("输入创意描述 (输入 'quit' 退出):\n")

        for name, agent in self.agents.items():
            asyncio.create_task(agent.run_loop())
        await asyncio.sleep(0.1)

        director = self.agents["Director"]

        try:
            prompt = input("创意 > ").strip()
            if prompt.lower() in ("quit", "exit", "q"):
                return
            if not prompt:
                prompt = "一个机器人在末日废墟中寻找一朵花"

            print(f"\n开始制作: {prompt}\n")
            result = await director.handle_task({
                "prompt": prompt,
                "genre": "sci-fi",
                "duration_hint": 60,
                "quality_threshold": 0.80,
            })

            summary = self._build_summary(result)
            print(f"\n完成!")
            print(f"   标题: {summary['title']}")
            print(f"   镜头: {summary['stats']['shots_completed']}/{summary['stats']['shots_planned']}")
            print(f"   时长: {summary['stats']['total_duration']:.1f}s")
            print(f"   输出: {summary['output_video'] or '(mock 模式, 无真实输出)'}")
            print(f"   详情: {self.output_dir / 'summary.json'}")

        except (KeyboardInterrupt, EOFError):
            print("\n退出")

        finally:
            for name in self.agents:
                await self.bus.send("system", name, "shutdown", {})
            await asyncio.sleep(0.2)
            await asset_db.close()
