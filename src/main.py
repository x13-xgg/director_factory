"""导演工厂 — 生产入口 (CLI + API 服务启动)"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import config
from src.core.logging import get_logger
from src.pipeline.runner import PipelineRunner

log = get_logger("Main")


async def run_pipeline(prompt: str, **kwargs):
    """运行完整管线"""
    runner = PipelineRunner(output_dir=kwargs.pop("output_dir", config.output_dir))
    result = await runner.run(prompt, **kwargs)
    return result


async def start_metrics_server():
    """启动 Prometheus metrics 端点"""
    if not config.logging.metrics_enabled:
        return
    try:
        from prometheus_client import start_http_server
        port = config.logging.metrics_port
        start_http_server(port)
        log.info(f"Metrics 服务已启动: http://0.0.0.0:{port}")
    except ImportError:
        log.warn("prometheus_client 未安装，metrics 不可用")


def main():
    parser = argparse.ArgumentParser(description="全自动导演工厂")
    sub = parser.add_subparsers(dest="command")

    # run — 运行管线
    run_parser = sub.add_parser("run", help="运行视频生产管线")
    run_parser.add_argument("prompt", help="创意描述")
    run_parser.add_argument("--genre", default="drama")
    run_parser.add_argument("--duration", type=int, default=60, help="目标时长 (秒)")
    run_parser.add_argument("--style", default="", help="风格参考")
    run_parser.add_argument("--quality", type=float, default=0.85, help="质量阈值")
    run_parser.add_argument("--language", default="zh", choices=["zh", "en", "ja", "ko"], help="目标语言")

    # resume — 从检查点恢复
    resume_parser = sub.add_parser("resume", help="从检查点恢复")
    resume_parser.add_argument("project_id", help="项目 ID")

    # retry — 增量重试失败镜头
    retry_parser = sub.add_parser("retry", help="增量重试失败镜头")
    retry_parser.add_argument("project_id", help="项目 ID")

    # serve — API 服务
    serve_parser = sub.add_parser("serve", help="启动 API 服务")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)

    # interactive — 交互模式
    sub.add_parser("interactive", help="交互模式")

    args = parser.parse_args()

    if args.command == "run":
        result = asyncio.run(run_pipeline(
            args.prompt,
            genre=args.genre,
            duration_hint=args.duration,
            style_ref=args.style,
            quality_threshold=args.quality,
            language=args.language,
        ))
        print(f"完成: {result.get('title', 'Unknown')}")
        print(f"状态: {result.get('status', 'unknown')}")
        sys.exit(0 if result.get("status") == "ok" else 1)

    elif args.command == "resume":
        runner = PipelineRunner()
        result = asyncio.run(runner.resume(args.project_id))
        print(f"恢复完成: {result.get('title', 'Unknown')}")
        sys.exit(0 if result.get("status") == "ok" else 1)

    elif args.command == "retry":
        runner = PipelineRunner()
        result = asyncio.run(runner.retry_failed(args.project_id))
        print(f"重试完成: {result.get('status', 'unknown')}")
        sys.exit(0 if result.get("status") == "ok" else 1)

    elif args.command == "serve":
        asyncio.run(start_metrics_server())
        import uvicorn
        from src.api import create_app
        app = create_app()
        uvicorn.run(app, host=args.host, port=args.port)

    elif args.command == "interactive":
        runner = PipelineRunner()
        asyncio.run(runner.run_interactive())

    else:
        # 默认: 交互模式
        runner = PipelineRunner()
        asyncio.run(runner.run_interactive())


if __name__ == "__main__":
    main()
