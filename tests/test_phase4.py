"""Phase 4 测试 — 并行调度优化 + Prompt 缓存 + GPU 调度 + 检查点/恢复"""

import asyncio
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.base import ToolRegistry
from src.tools.text_gen import TextGenTool
from src.tools.performance import (
    PromptCacheTool,
    GPUSchedulerTool,
    CheckpointTool,
)
from src.tools.asset_db import asset_db
from src.core.agent import AgentConfig
from src.core.message_bus import MessageBus
from src.agents.scheduler import SchedulerAgent


# ── 工具级测试 ────────────────────────────────────────

async def test_prompt_cache_tool():
    """测试 Prompt 缓存 — 存储与检索"""
    tool = PromptCacheTool()
    registry = ToolRegistry()
    registry.register(tool)

    shot_spec = {
        "framing": "medium",
        "emotion": "hope",
        "scene_id": "scene_01",
        "action_description": "机器人发现一朵花",
        "characters_in_frame": ["robot_01"],
    }

    # 存储
    store_result = await registry.call("prompt_cache", {
        "shot_spec": shot_spec,
        "prompt": "A rusty robot finding a glowing flower in ruins, cinematic, photorealistic, medium shot",
        "negative_prompt": "blurry, deformed, low quality",
        "params": {"steps": 30, "cfg": 7.5},
        "quality_score": 0.92,
    })
    assert store_result.status == "ok"
    assert store_result.data.get("cached") is True

    # 检索 — 精确命中
    retrieve_result = await registry.call("prompt_cache", {
        "shot_spec": shot_spec,
    })
    assert retrieve_result.status == "ok"
    data = retrieve_result.data
    assert data.get("hit") is True
    assert data.get("match_type") == "exact"
    assert "robot" in data.get("prompt", "")

    # 检索 — 相似匹配
    similar_spec = dict(shot_spec)
    similar_spec["action_description"] = "机器人找到花朵"
    similar_result = await registry.call("prompt_cache", {
        "shot_spec": similar_spec,
    })
    assert similar_result.data.get("hit") is True

    # 检索 — 未命中
    unknown_spec = {
        "framing": "extreme_wide",
        "emotion": "anger",
        "scene_id": "space_battle",
        "action_description": "星舰爆炸",
        "characters_in_frame": ["alien_01"],
    }
    miss_result = await registry.call("prompt_cache", {
        "shot_spec": unknown_spec,
    })
    assert miss_result.data.get("hit") is False

    # 低质量不缓存
    low_q_result = await registry.call("prompt_cache", {
        "shot_spec": unknown_spec,
        "prompt": "bad prompt",
        "quality_score": 0.70,
    })
    assert low_q_result.data.get("cached") is not True

    # 统计
    stats = tool.get_stats()
    assert stats["hits"] >= 2
    assert stats["misses"] >= 1
    print(f"    Hit rate: {stats['hit_rate_pct']:.0f}%, entries: {stats['cache_entries']}")
    print("  P test_prompt_cache_tool")


async def test_gpu_scheduler_tool():
    """测试 GPU 资源调度"""
    tool = GPUSchedulerTool(total_vram_gb=24.0, max_concurrent=4)
    registry = ToolRegistry()
    registry.register(tool)

    # 请求 GPU 资源
    r1 = await registry.call("gpu_scheduler", {
        "action": "request",
        "job_id": "shot_01",
        "vram_required_gb": 2.5,
        "priority": 1,
    })
    assert r1.data.get("allocated") is True
    assert r1.data.get("status") == "running"

    r2 = await registry.call("gpu_scheduler", {
        "action": "request",
        "job_id": "shot_02",
        "vram_required_gb": 2.5,
    })
    assert r2.data.get("allocated") is True

    r3 = await registry.call("gpu_scheduler", {
        "action": "request",
        "job_id": "shot_03",
        "vram_required_gb": 2.5,
    })
    assert r3.data.get("allocated") is True

    r4 = await registry.call("gpu_scheduler", {
        "action": "request",
        "job_id": "shot_04",
        "vram_required_gb": 2.5,
    })
    assert r4.data.get("allocated") is True

    # 第5个应该排队 (max_concurrent=4)
    r5 = await registry.call("gpu_scheduler", {
        "action": "request",
        "job_id": "shot_05",
        "vram_required_gb": 2.5,
    })
    assert r5.data.get("allocated") is False
    assert r5.data.get("status") == "queued"
    assert r5.data.get("queue_position") == 1

    # 释放一个 + 验证队列启动
    release_result = await registry.call("gpu_scheduler", {
        "action": "release",
        "job_id": "shot_01",
    })
    assert release_result.data.get("released", {}).get("job_id") == "shot_01"
    assert "shot_05" in release_result.data.get("started_from_queue", [])

    # 状态
    status = await registry.call("gpu_scheduler", {"action": "status"})
    assert status.data.get("active_jobs") == 4
    assert status.data.get("queue_length") == 0
    print(f"    VRAM: {status.data['used_vram_gb']:.1f}/{status.data['total_vram_gb']}GB, util={status.data['utilization_pct']:.0f}%")
    print("  P test_gpu_scheduler_tool")


async def test_checkpoint_tool():
    """测试检查点 — 保存/加载/列表"""
    tool = CheckpointTool(default_dir="outputs/test_checkpoints")
    registry = ToolRegistry()
    registry.register(tool)

    project_id = "test_proj_phase4"

    # 保存
    save_state = {
        "screenplay": {"title": "Test"},
        "shotlist": {"shots": [], "total_duration": 60},
        "style_guide": {"visual_mood": "cinematic"},
        "char_profiles": {"robot_01": {}},
        "completed_shots": ["shot_01", "shot_02"],
        "failed_shots": ["shot_03"],
        "frames": [{"shot_id": "shot_01"}, {"shot_id": "shot_02"}],
    }
    save_result = await registry.call("checkpoint", {
        "action": "save",
        "project_id": project_id,
        "state": save_state,
    })
    assert save_result.status == "ok"

    # 加载
    load_result = await registry.call("checkpoint", {
        "action": "load",
        "project_id": project_id,
    })
    assert load_result.data.get("found") is True
    assert load_result.data.get("completed_shot_count") == 2
    assert load_result.data.get("failed_shot_count") == 1
    assert "creative" in load_result.data.get("completed_phases", [])

    # 列表
    list_result = await registry.call("checkpoint", {"action": "list"})
    assert list_result.data.get("count", 0) >= 1

    # 不存在
    missing = await registry.call("checkpoint", {
        "action": "load",
        "project_id": "nonexistent",
    })
    assert missing.data.get("found") is False

    print(f"    Checkpoint: {save_result.data['checkpoint_path']}, phases={load_result.data['completed_phases']}")
    print("  P test_checkpoint_tool")


# ── Scheduler 增强测试 ─────────────────────────────

async def test_scheduler_enhanced_parallel():
    """测试增强调度器 — 4 条件并行策略"""
    bus = MessageBus()
    tools = ToolRegistry()

    config = AgentConfig(name="Scheduler", role="scheduler")
    agent = SchedulerAgent(config, bus, tools)
    await agent.start()

    # 构建多场景 + 多角色 + 有依赖的镜头列表
    shots = [
        {"id": "s1", "scene_id": "ruins", "characters_in_frame": ["robot_01"], "dependencies": []},
        {"id": "s2", "scene_id": "ruins", "characters_in_frame": ["robot_01"], "dependencies": ["s1"]},
        {"id": "s3", "scene_id": "ruins", "characters_in_frame": ["robot_02"], "dependencies": []},
        {"id": "s4", "scene_id": "forest", "characters_in_frame": ["robot_01"], "dependencies": []},
        {"id": "s5", "scene_id": "forest", "characters_in_frame": ["robot_02"], "dependencies": []},
        {"id": "s6", "scene_id": "factory", "characters_in_frame": ["robot_01", "robot_02"], "dependencies": []},
    ]

    init_result = await agent.handle_task({
        "action": "init",
        "shotlist": {"shots": shots},
        "max_retries": 3,
    })
    assert init_result["status"] == "ok"
    assert init_result["total_shots"] == 6
    assert init_result["max_concurrent"] >= 8

    # 依赖图
    dep_result = await agent.handle_task({"action": "dep_graph"})
    deps = dep_result.get("dep_graph", {})
    assert "s2" in deps.get("dependencies", {})
    assert "s1" in deps["dependencies"]["s2"]

    # 第一批: s1, s3, s4, s5, s6 应该可用 (s2 依赖 s1)
    batch1 = await agent.handle_task({"action": "next_batch"})
    batch1_ids = batch1.get("batch", [])
    assert "s1" in batch1_ids
    assert "s3" in batch1_ids
    assert "s4" in batch1_ids  # different scene
    assert "s2" not in batch1_ids  # blocked by s1

    # 验证场景多样性
    scenes_in_batch = set()
    for sid in batch1_ids:
        shot = next(s for s in shots if s["id"] == sid)
        scenes_in_batch.add(shot["scene_id"])
    assert len(scenes_in_batch) >= 2

    # 标记 s1 完成 → 解锁 s2
    mark = await agent.handle_task({"action": "mark_done", "shot_id": "s1"})
    assert "s2" in mark.get("unblocked", [])

    # 标记 s3 完成
    await agent.handle_task({"action": "mark_done", "shot_id": "s3"})

    # 第一批的剩下标记完成
    for sid in batch1_ids:
        if sid not in ("s1", "s3"):
            await agent.handle_task({"action": "mark_done", "shot_id": sid})

    # 下一批应包含 s2
    batch2 = await agent.handle_task({"action": "next_batch"})
    assert "s2" in batch2.get("batch", [])

    # 统计
    stats = await agent.handle_task({"action": "stats"})
    assert stats["stats"]["completed"] == 5

    print(f"    Batch1: {batch1_ids}, scenes={batch1.get('scene_diversity', 0)}")
    print(f"    Dep graph: {dep_result.get('dep_graph', {}).get('dependencies', {})}")
    print("  P test_scheduler_enhanced_parallel")


async def test_scheduler_stats_blocked():
    """测试调度统计与阻塞分析"""
    bus = MessageBus()
    tools = ToolRegistry()

    config = AgentConfig(name="Scheduler", role="scheduler")
    agent = SchedulerAgent(config, bus, tools)
    await agent.start()

    shots = [
        {"id": "a1", "scene_id": "room", "characters_in_frame": ["hero"], "dependencies": []},
        {"id": "a2", "scene_id": "room", "characters_in_frame": ["hero"], "dependencies": ["a1"]},
        {"id": "a3", "scene_id": "room", "characters_in_frame": ["hero"], "dependencies": ["a2"]},
    ]

    await agent.handle_task({"action": "init", "shotlist": {"shots": shots}})

    # 第一批只有 a1
    batch1 = await agent.handle_task({"action": "next_batch"})
    assert batch1.get("batch") == ["a1"]

    # 阻塞分析
    blocked = await agent.handle_task({"action": "blocked_shots"})
    blocked_shots = blocked.get("blocked_shots", [])
    assert len(blocked_shots) == 2

    # 统计
    stats = await agent.handle_task({"action": "stats"})
    assert stats["stats"]["blocked_currently"] == 2
    assert stats["stats"]["completed"] == 0
    assert stats["stats"]["progress_pct"] == 0.0

    # 逐步完成
    await agent.handle_task({"action": "mark_done", "shot_id": "a1"})
    await agent.handle_task({"action": "mark_done", "shot_id": "a2"})
    await agent.handle_task({"action": "mark_done", "shot_id": "a3"})
    assert (await agent.handle_task({"action": "all_done"})).get("all_done") is True

    print(f"    Blocked shots: {blocked_shots}")
    print("  P test_scheduler_stats_blocked")


async def test_scheduler_retry_exhausted():
    """测试重试耗尽逻辑"""
    bus = MessageBus()
    tools = ToolRegistry()

    config = AgentConfig(name="Scheduler", role="scheduler")
    agent = SchedulerAgent(config, bus, tools)
    await agent.start()

    shots = [{"id": "r1", "scene_id": "test", "characters_in_frame": [], "dependencies": []}]
    await agent.handle_task({"action": "init", "shotlist": {"shots": shots}, "max_retries": 2})

    # 重试两次
    r1 = await agent.handle_task({"action": "retry", "shot_id": "r1", "feedback": "bad composition"})
    assert r1["status"] == "ok"
    r2 = await agent.handle_task({"action": "retry", "shot_id": "r1", "feedback": "still bad"})
    assert r2["status"] == "retry_exhausted"

    # exhausted 后不再出现在批处理中
    stats = await agent.handle_task({"action": "stats"})
    assert stats["stats"]["retries_exhausted"] == 1

    # mark_done on exhausted shot should be handled gracefully
    done_result = await agent.handle_task({"action": "mark_done", "shot_id": "r1"})
    stats2 = await agent.handle_task({"action": "stats"})
    # mark_done on exhausted shot still counts as completed
    assert stats2["stats"]["completed"] >= 1

    print(f"    Retry exhausted: {r2['retry_count']}/2")
    print("  P test_scheduler_retry_exhausted")


async def test_phase4_integration():
    """Phase 4 集成测试 — 调度 + 缓存 + GPU + 检查点协同"""
    print("\n  -- Phase 4 集成测试 --")

    bus = MessageBus()
    tools = ToolRegistry()
    tools.register(TextGenTool())
    tools.register(PromptCacheTool())
    tools.register(GPUSchedulerTool(total_vram_gb=24.0, max_concurrent=4))
    tools.register(CheckpointTool(default_dir="outputs/test_phase4_cp"))

    config = AgentConfig(name="Scheduler", role="scheduler")
    scheduler = SchedulerAgent(config, bus, tools)
    await scheduler.start()

    # 模拟完整调度流程
    shots = []
    for i in range(10):
        scene_id = f"scene_{i // 3}"
        shots.append({
            "id": f"p4_shot_{i:02d}",
            "scene_id": scene_id,
            "characters_in_frame": [f"char_{i % 3}"],
            "dependencies": [f"p4_shot_{i - 1:02d}"] if i % 3 == 0 and i > 0 else [],
        })

    await scheduler.handle_task({"action": "init", "shotlist": {"shots": shots}, "max_retries": 3})

    # 模拟生产循环
    completed = 0
    total_batches = 0
    gpu_tool = GPUSchedulerTool(total_vram_gb=24.0, max_concurrent=4)

    while completed < len(shots):
        batch_result = await scheduler.handle_task({"action": "next_batch"})
        batch = batch_result.get("batch", [])
        if not batch:
            blocked = await scheduler.handle_task({"action": "blocked_shots"})
            if not blocked.get("blocked_shots"):
                break
            # 检查被阻塞的镜头，标记那些依赖已满足的
            break  # 在真实管线中会等待或处理

        total_batches += 1

        # GPU 资源请求 (同步方法, 不 await)
        for sid in batch:
            gpu_tool._request_vram({"job_id": sid, "vram_required_gb": 2.5})

        # 模拟拍摄 → 完成
        for sid in batch:
            await scheduler.handle_task({"action": "mark_done", "shot_id": sid})
            gpu_tool._release_vram({"job_id": sid})
            completed += 1

    # 验证统计
    stats = await scheduler.handle_task({"action": "stats"})
    assert stats["stats"]["completed"] == len(shots)
    assert stats["stats"]["total_batches"] >= 2  # 至少分批

    # 检查点保存 — 直接使用 CheckpointTool
    cp_tool = CheckpointTool(default_dir="outputs/test_phase4_cp")
    from src.tools.base import ToolCall as TC
    save_call = TC(tool="checkpoint", params={
        "action": "save",
        "project_id": "phase4_integration",
        "state": {
            "screenplay": {"title": "Test"},
            "shotlist": {"shots": shots},
            "completed_shots": [s["id"] for s in shots],
            "failed_shots": [],
            "frames": [],
            "stats": stats["stats"],
        },
    }, caller="test")
    save_result = await cp_tool.execute(save_call)
    assert save_result.status == "ok"

    # Prompt 缓存验证
    cache_tool = PromptCacheTool()
    for i in range(5):
        store_call = TC(tool="prompt_cache", params={
            "shot_spec": {
                "framing": "medium",
                "emotion": "neutral",
                "scene_id": f"scene_{i // 2}",
                "action_description": f"action {i}",
                "characters_in_frame": [f"char_{i % 3}"],
            },
            "prompt": f"Test prompt for shot {i}, high quality cinematic",
            "quality_score": 0.88 + (i % 3) * 0.03,
        }, caller="test")
        await cache_tool.execute(store_call)

    cache_stats = cache_tool.get_stats()
    assert cache_stats["cache_entries"] >= 5

    # 验证 GPU 资源已全部释放
    gpu_status = gpu_tool._get_status()
    gpu_data = gpu_status.data
    assert gpu_data["active_jobs"] == 0
    assert gpu_data["available_vram_gb"] > 20

    print(f"    完成镜头: {completed}, 批次数: {total_batches}")
    print(f"    Batch 统计: avg={stats['stats']['avg_batch_size']}, max={stats['stats']['max_batch_size']}")
    print(f"    GPU 利用率: {stats['stats']['gpu_utilization']:.0f}%")
    print(f"    Prompt 缓存: {cache_stats['cache_entries']} 条, hit rate={cache_stats['hit_rate_pct']:.0f}%")
    print("  P test_phase4_integration")


async def run_all():
    print("\n" + "=" * 60)
    print("Phase 4 - Parallel & Performance Tests")
    print("=" * 60)

    tests = [
        ("PromptCacheTool", test_prompt_cache_tool),
        ("GPUSchedulerTool", test_gpu_scheduler_tool),
        ("CheckpointTool", test_checkpoint_tool),
        ("Scheduler Enhanced Parallel", test_scheduler_enhanced_parallel),
        ("Scheduler Stats & Blocked", test_scheduler_stats_blocked),
        ("Scheduler Retry Exhausted", test_scheduler_retry_exhausted),
        ("Phase4 Integration", test_phase4_integration),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            print(f"\n-- {name} --")
            await fn()
            passed += 1
        except Exception as e:
            print(f"  X {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Result: {passed} passed, {failed} failed")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all())
    sys.exit(0 if success else 1)
