#!/usr/bin/env python3
"""全自动导演工厂 — CLI 入口

用法:
  python main.py "一个机器人在末日废墟中寻找一朵花"
  python main.py --genre sci-fi --duration 120 "创意描述"
  python main.py --interactive
  python main.py --demo              # 运行演示
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# 确保项目路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipeline.runner import PipelineRunner


async def main():
    parser = argparse.ArgumentParser(
        description="🎬 全自动导演工厂 — 输入创意，输出成片",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py "废墟中寻找花朵的机器人"
  python main.py --genre sci-fi --duration 90 "两个AI在月球上的对话"
  python main.py --interactive
  python main.py --demo
        """,
    )
    parser.add_argument("prompt", nargs="?", help="创意描述 (一句话/一段文本)")
    parser.add_argument("--genre", default="drama", help="类型 (默认: drama)")
    parser.add_argument("--duration", type=int, default=60, help="目标时长秒数 (默认: 60)")
    parser.add_argument("--style", default="", help="风格参考")
    parser.add_argument("--quality", type=float, default=0.85, help="质量阈值 0-1 (默认: 0.85)")
    parser.add_argument("--output", default="outputs", help="输出目录 (默认: outputs)")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    parser.add_argument("--demo", "-d", action="store_true", help="运行内置演示")
    parser.add_argument("--mock", action="store_true", help="强制 mock 模式 (无需 API key)")

    args = parser.parse_args()

    runner = PipelineRunner(output_dir=args.output)

    if args.interactive:
        await runner.run_interactive()
        return

    if args.demo:
        prompt = "一个锈迹斑斑的机器人在末日废墟中寻找最后一朵盛开的花，色调冷峻，风格写实"
        print(f"\n🎬 演示模式")
        print(f"   创意: {prompt}")
        print(f"   类型: sci-fi")
        print(f"   时长: 60s\n")
        print("=" * 60)
        t0 = time.time()
        result = await runner.run(
            prompt,
            genre="sci-fi",
            duration_hint=60,
            style_ref="blade-runner-meets-wall-e",
            quality_threshold=0.80,
        )
        elapsed = time.time() - t0
        _print_result(result, elapsed)
        return

    if args.prompt:
        print(f"\n🎬 创意: {args.prompt}")
        print(f"   类型: {args.genre}")
        print(f"   时长: {args.duration}s\n")
        print("=" * 60)
        t0 = time.time()
        result = await runner.run(
            args.prompt,
            genre=args.genre,
            duration_hint=args.duration,
            style_ref=args.style,
            quality_threshold=args.quality,
        )
        elapsed = time.time() - t0
        _print_result(result, elapsed)
        return

    # 无参数 → 显示帮助
    parser.print_help()


def _print_result(result: dict, elapsed: float):
    stats = result.get("stats", {})
    print("\n" + "=" * 60)
    print("📊 管线执行结果")
    print("=" * 60)
    print(f"  标题:       {result.get('title', 'Untitled')}")
    print(f"  状态:       {result.get('status', 'unknown')}")
    print(f"  角色数:     {stats.get('characters', 0)}")
    print(f"  场景数:     {stats.get('scenes', 0)}")
    print(f"  计划镜头:   {stats.get('shots_planned', 0)}")
    print(f"  完成镜头:   {stats.get('shots_completed', 0)}")
    print(f"  总时长:     {stats.get('total_duration', 0):.1f}s")
    print(f"  输出文件:   {result.get('output_video', '(mock)')}")
    print(f"  耗时:       {elapsed:.1f}s")
    print()


if __name__ == "__main__":
    asyncio.run(main())
