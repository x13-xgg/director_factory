"""一次性脚本：用 project_1782876447 的原始 prompt 重跑管线"""
import asyncio
import sys
import os
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.runner import PipelineRunner
from src.core.config import config

PROMPT = (
    "卡卡卡，嗒嗒……"
    "一双灵巧的手飞舞着操纵着键盘和鼠标，富有节奏的敲击声仿佛是一首轻快的乐章。"
    "屏幕中漫天的光华闪过，对手飞扬着血花倒了下去。"
    "呵呵。"
)

async def main():
    runner = PipelineRunner(output_dir="outputs/project_1782876447")
    result = await runner.run(
        PROMPT,
        genre="animation",
        duration_hint=45,
        style_ref="",
        quality_threshold=0.80,
        language="zh",
    )
    print(f"Title: {result.get('title', 'Unknown')}")
    print(f"Status: {result.get('status', 'unknown')}")
    stats = result.get('stats', {})
    print(f"Shots: {stats.get('shots_completed', 0)}/{stats.get('shots_planned', 0)}")
    print(f"Duration: {stats.get('total_duration', 0)}s")
    frames = result.get('frames', [])
    print(f"Frames: {len(frames)}")

if __name__ == "__main__":
    asyncio.run(main())
