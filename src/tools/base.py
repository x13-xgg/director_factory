"""工具调用协议 — ToolCall / ToolResult / BaseTool"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolCall:
    tool: str
    params: dict
    caller: str = ""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])


@dataclass
class ToolResult:
    status: str  # "ok" | "retry" | "fail"
    data: Any = None
    metadata: dict = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)
    trace_id: str = ""

    @classmethod
    def ok(cls, data: Any = None, **metadata) -> ToolResult:
        return cls(status="ok", data=data, metadata=metadata)

    @classmethod
    def retry(cls, data: Any = None, suggestions: list[str] | None = None, **metadata) -> ToolResult:
        return cls(status="retry", data=data, suggestions=suggestions or [], metadata=metadata)

    @classmethod
    def fail(cls, data: Any = None, suggestions: list[str] | None = None, **metadata) -> ToolResult:
        return cls(status="fail", data=data, suggestions=suggestions or [], metadata=metadata)


class BaseTool(ABC):
    """工具基类 — 所有工具继承此类"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def execute(self, call: ToolCall) -> ToolResult:
        ...

    @abstractmethod
    def schema(self) -> dict:
        """返回此工具的 JSON Schema，供 Agent 自动发现"""
        ...


class ToolRegistry:
    """工具注册中心"""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._hooks: dict[str, list[Callable]] = {}

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def list_schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    async def call(self, tool_name: str, params: dict, caller: str = "") -> ToolResult:
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult.fail(data=None, suggestions=[f"Tool '{tool_name}' not found. Available: {self.list_tools()}"])
        call = ToolCall(tool=tool_name, params=params, caller=caller)
        return await tool.execute(call)

    def on(self, event: str, handler: Callable):
        self._hooks.setdefault(event, []).append(handler)

    async def emit(self, event: str, data: dict):
        for h in self._hooks.get(event, []):
            await h(data)
