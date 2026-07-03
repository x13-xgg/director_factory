"""统一日志与追踪 (OpenTelemetry 风格)"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


@dataclass
class TraceSpan:
    trace_id: str
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: str = ""
    name: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    status: str = "ok"
    metadata: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


class Tracer:
    """简易追踪器"""

    def __init__(self):
        self._spans: list[TraceSpan] = []
        self._active: dict[str, TraceSpan] = {}

    def start_span(self, name: str, parent_id: str = "", trace_id: str = "") -> str:
        tid = trace_id or str(uuid.uuid4())[:16]
        span = TraceSpan(
            trace_id=tid,
            parent_id=parent_id,
            name=name,
            start_time=time.time(),
        )
        self._active[span.span_id] = span
        return span.span_id

    def end_span(self, span_id: str, status: str = "ok", metadata: dict | None = None):
        span = self._active.pop(span_id, None)
        if span:
            span.end_time = time.time()
            span.status = status
            if metadata:
                span.metadata.update(metadata)
            self._spans.append(span)

    def get_trace(self, trace_id: str) -> list[TraceSpan]:
        return [s for s in self._spans if s.trace_id == trace_id]

    def flush(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        records = []
        for s in self._spans:
            records.append({
                "trace_id": s.trace_id,
                "span_id": s.span_id,
                "parent_id": s.parent_id,
                "name": s.name,
                "duration_ms": round(s.duration_ms, 2),
                "status": s.status,
                "metadata": s.metadata,
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)


# 全局单例
tracer = Tracer()


class Logger:
    def __init__(self, name: str):
        self.name = name

    def _log(self, level: LogLevel, msg: str, **kwargs):
        ts = time.strftime("%H:%M:%S")
        extra = " " + json.dumps(kwargs, ensure_ascii=False) if kwargs else ""
        line = f"[{level.value.upper():5s}] [{ts}] [{self.name}] {msg}{extra}"
        # 编码安全输出 (Windows GBK 终端兼容)
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            print(line.encode("utf-8", errors="replace").decode("utf-8", errors="replace"), flush=True)

    def debug(self, msg: str, **kwargs):
        self._log(LogLevel.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs):
        self._log(LogLevel.INFO, msg, **kwargs)

    def warn(self, msg: str, **kwargs):
        self._log(LogLevel.WARN, msg, **kwargs)

    def error(self, msg: str, **kwargs):
        self._log(LogLevel.ERROR, msg, **kwargs)


def get_logger(name: str) -> Logger:
    return Logger(name)
