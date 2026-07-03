"""Agent 基类 — 所有 Agent 的抽象基类"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.core.message_bus import MessageBus, Message
from src.core.logging import get_logger, tracer
from src.tools.base import ToolRegistry, ToolResult


@dataclass
class AgentConfig:
    name: str
    role: str
    model: str = "claude-sonnet-4-6"
    max_retries: int = 3
    temperature: float = 0.7
    tools: list[str] = field(default_factory=list)


class DecisionLevel:
    """Agent 决策权分级"""
    L0_AUTONOMOUS = "L0"   # 自主决策
    L1_REPORT = "L1"       # 决策后通知总监
    L2_APPROVE = "L2"      # 提案待审批
    L3_INSTRUCT = "L3"     # 按总监指令执行
    L4_ARBITRATE = "L4"    # 总监裁决


class BaseAgent(ABC):
    """Agent 基类"""

    def __init__(self, config: AgentConfig, bus: MessageBus, tools: ToolRegistry):
        self.config = config
        self.bus = bus
        self.tools = tools
        self.log = get_logger(f"Agent.{config.name}")
        self.state: dict[str, Any] = {}

    # ── 抽象方法 ───────────────────────────────────

    @abstractmethod
    async def handle_task(self, task: dict) -> dict:
        """Agent 主入口，接收任务返回结果"""
        ...

    # ── 工具调用 ───────────────────────────────────

    async def call_tool(self, tool_name: str, params: dict) -> ToolResult:
        span_id = tracer.start_span(f"tool:{tool_name}", parent_id="")
        self.log.debug(f"调用工具: {tool_name}", params=params)
        result = await self.tools.call(tool_name, params, caller=self.config.name)
        tracer.end_span(span_id, status=result.status, metadata={"tool": tool_name})
        if result.suggestions:
            self.log.info(f"工具建议: {result.suggestions}")
        return result

    # ── 消息通信 ───────────────────────────────────

    async def send_to(self, target: str, msg_type: str, payload: dict):
        await self.bus.send(self.config.name, target, msg_type, payload)

    async def broadcast(self, msg_type: str, payload: dict):
        await self.bus.broadcast(self.config.name, msg_type, payload)

    async def request(self, target: str, msg_type: str, payload: dict, timeout: float = 60.0) -> dict | None:
        return await self.bus.request(self.config.name, target, msg_type, payload, timeout)

    async def receive(self, timeout: float = 5.0) -> Message | None:
        return await self.bus.receive(self.config.name, timeout)

    # ── 总监汇报 ───────────────────────────────────

    async def report(self, status: str, data: dict):
        await self.send_to("Director", "status_report", {
            "agent": self.config.name,
            "status": status,
            "data": data,
        })

    # ── 委托子 Agent ───────────────────────────────

    async def delegate(self, target: str, task: dict, timeout: float = 120.0) -> dict:
        """委托任务给其他 Agent，等待结果"""
        span_id = tracer.start_span(f"delegate:{target}")
        self.log.info(f"委托 {target}: {json.dumps(task, ensure_ascii=False)[:200]}")
        result = await self.request(target, "task", task, timeout=timeout)
        if result is None:
            tracer.end_span(span_id, status="timeout")
            return {"status": "error", "error": f"Agent '{target}' timeout"}
        tracer.end_span(span_id, status="ok")
        return result

    # ── 质量评估 ───────────────────────────────────

    async def evaluate(self, data: dict) -> float:
        """评估产出质量，返回 0-1 分数"""
        scores = data.get("scores", {})
        if not scores:
            return 1.0  # 无评分数据默认通过
        weights = {"composition": 0.40, "consistency": 0.35, "light": 0.15, "emotion": 0.10}
        total = sum(weights.get(k, 0) * v for k, v in scores.items())
        return min(total, 1.0)

    # ── 审核-修改循环 ──────────────────────────────

    async def review_loop(self, draft: Any, evaluator, max_rounds: int = 3) -> Any:
        """通用审核-修改循环"""
        for attempt in range(max_rounds):
            score = await evaluator(draft)
            if score >= 0.85:
                return draft
            self.log.info(f"审核未通过 (score={score:.2f}, round={attempt + 1})，进入修订...")
            draft = await self.revise(draft, f"Quality score {score:.2f} below threshold 0.85")
        return draft

    async def revise(self, draft: Any, feedback: str) -> Any:
        """子类可覆盖此方法实现修订逻辑"""
        return draft

    # ── 生命周期 ───────────────────────────────────

    async def start(self):
        """Agent 启动 (子类可覆盖)"""
        await self.bus.register_agent(self.config.name)
        self.log.info(f"Agent 启动: {self.config.name} (role={self.config.role})")

    async def stop(self):
        """Agent 停止 (子类可覆盖)"""
        self.log.info(f"Agent 停止: {self.config.name}")

    async def run_loop(self):
        """主消息循环 (可选)"""
        await self.start()
        try:
            while True:
                try:
                    msg = await self.receive(timeout=10.0)
                except Exception as e:
                    self.log.warn(f"接收消息异常: {e}, 重试中...")
                    await asyncio.sleep(1)
                    continue
                if msg is None:
                    continue
                try:
                    if msg.msg_type == "task":
                        result = await self.handle_task(msg.payload)
                        await self.bus.reply(msg, result)
                    elif msg.msg_type == "shutdown":
                        break
                except Exception as e:
                    self.log.error(f"处理任务异常: {e}")
                    await self.bus.reply(msg, {"status": "error", "error": str(e)})
        finally:
            await self.stop()
