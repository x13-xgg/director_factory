"""Phase 5 测试 — 质量增强 + 跨模态审核 + 大规模压力测试 + 性能基准"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.base import ToolRegistry
from src.tools.scorers import (
    CompositionScorerTool,
    RhythmScorerTool,
    EmotionAlignmentCheckerTool,
    QualityAggregatorTool,
    BenchmarkTool,
)


# ── 增强审核器测试 ────────────────────────────────────

async def test_rhythm_scorer_enhanced():
    """测试增强节奏评分 — 场景 pacing + 转场质量"""
    tool = RhythmScorerTool()
    registry = ToolRegistry()
    registry.register(tool)

    timeline = {
        "clips": [
            {"duration": 3.0, "shot_id": "s1"},
            {"duration": 2.5, "shot_id": "s2"},
            {"duration": 1.8, "shot_id": "s3"},
            {"duration": 5.5, "shot_id": "s4"},
            {"duration": 2.0, "shot_id": "s5"},
            {"duration": 1.2, "shot_id": "s6"},
        ],
    }
    shotlist = {
        "shots": [
            {"id": "s1", "scene_id": "scene_a", "emotion": "neutral", "camera": {"framing": "wide"}},
            {"id": "s2", "scene_id": "scene_a", "emotion": "tension", "camera": {"framing": "medium"}},
            {"id": "s3", "scene_id": "scene_a", "emotion": "tension", "camera": {"framing": "close_up"}},
            {"id": "s4", "scene_id": "scene_b", "emotion": "sadness", "camera": {"framing": "wide"}},
            {"id": "s5", "scene_id": "scene_b", "emotion": "hope", "camera": {"framing": "medium"}},
            {"id": "s6", "scene_id": "scene_b", "emotion": "joy", "camera": {"framing": "close_up"}},
        ],
    }

    result = await registry.call("rhythm_scorer", {
        "timeline": timeline,
        "target_emotion_curve": ["neutral", "tension", "tension", "sadness", "hope", "joy"],
        "shotlist": shotlist,
    })

    assert result.status == "ok"
    data = result.data

    # shot 评分
    shot_scores = data.get("shot_scores", [])
    assert len(shot_scores) == 6
    for s in shot_scores:
        assert "score" in s
        assert "duration" in s
        assert "in_range" in s

    # 转场评分
    transitions = data.get("transition_scores", [])
    assert len(transitions) == 5  # 6 shots → 5 transitions
    for t in transitions:
        assert "transition_score" in t
        assert "emotion_change" in t

    # 场景 pacing
    scene_pacing = data.get("scene_pacing", [])
    assert len(scene_pacing) == 2  # scene_a, scene_b
    for sp in scene_pacing:
        assert "pacing_score" in sp

    assert data["pacing_match"] > 0
    assert data["duration_variance"] >= 0

    # 建议 (shot_3 = s4 过长)
    suggestions = data.get("suggestions", [])
    assert len(suggestions) >= 1  # at least the long shot

    print(f"    Pacing: {data['pacing_match']:.3f}, avg dur={data['avg_shot_duration']}s")
    for sp in scene_pacing:
        print(f"    Scene {sp['scene_id']}: pacing={sp['pacing_score']:.2f}")
    print(f"    Suggestions: {len(suggestions)}")
    print("  P test_rhythm_scorer_enhanced")


async def test_emotion_alignment_enhanced():
    """测试增强情绪对齐 — 跨模态 3-way + 转换自然度"""
    tool = EmotionAlignmentCheckerTool()
    registry = ToolRegistry()
    registry.register(tool)

    # 测试自然转换
    r1 = await registry.call("emotion_alignment_checker", {
        "image_path": "test.png",
        "audio_path": "test.wav",
        "target_emotion": "hope",
        "prev_emotion": "neutral",
        "next_emotion": "joy",
        "dialog_text": "我们会找到的",
    })
    assert r1.status == "ok"
    d1 = r1.data
    assert "visual_emotion" in d1
    assert "audio_emotion" in d1
    assert "text_emotion" in d1
    assert "alignment_matrix" in d1
    assert "transition_naturalness" in d1

    # neutral→hope 应该自然
    from_prev = d1["transition_naturalness"].get("from_prev", {})
    assert from_prev.get("natural") is True

    # hope→joy 应该自然
    to_next = d1["transition_naturalness"].get("to_next", {})
    assert to_next.get("natural") is True

    # 测试不自然转换
    r2 = await registry.call("emotion_alignment_checker", {
        "image_path": "test2.png",
        "audio_path": "test2.wav",
        "target_emotion": "anger",
        "prev_emotion": "serene",
        "next_emotion": "sadness",
        "dialog_text": "",
    })
    d2 = r2.data
    from_prev2 = d2["transition_naturalness"].get("from_prev", {})
    # serene→anger 不在自然转换中
    assert from_prev2.get("natural") is False

    print(f"    Hope: visual={d1['visual_emotion']['confidence']:.2f}, audio={d1['audio_emotion']['confidence']:.2f}")
    print(f"    Natural transitions: from_prev={from_prev['natural']}, to_next={to_next['natural']}")
    print(f"    Unnatural: serene→anger={from_prev2['natural']}")
    print("  P test_emotion_alignment_enhanced")


async def test_quality_aggregator_enhanced():
    """测试增强质量聚合 — 分维度阈值 + 建议排序"""
    tool = QualityAggregatorTool()
    registry = ToolRegistry()
    registry.register(tool)

    # 全部通过
    r1 = await registry.call("quality_aggregator", {
        "composition_score": 0.88,
        "consistency_score": 0.90,
        "light_score": 0.85,
        "emotion_score": 0.82,
        "shot_id": "good_shot",
    })
    assert r1.data["pass"] is True
    assert len(r1.data["threshold_failures"]) == 0

    # 分维度不达标但综合分通过
    r2 = await registry.call("quality_aggregator", {
        "composition_score": 0.90,
        "consistency_score": 0.92,
        "light_score": 0.68,  # 低于 0.70 阈值
        "emotion_score": 0.85,
        "shot_id": "dim_fail_shot",
    })
    assert r2.data["pass"] is False  # 分维度不通过
    assert r2.data["passed_by_overall"] is True  # 综合分可能通过
    assert len(r2.data["threshold_failures"]) >= 1

    # 综合分不达标
    r3 = await registry.call("quality_aggregator", {
        "composition_score": 0.72,
        "consistency_score": 0.75,
        "light_score": 0.78,
        "emotion_score": 0.80,
        "shot_id": "overall_fail_shot",
    })
    assert r3.data["pass"] is False

    # 建议排序
    suggestions = r3.data.get("suggestions", [])
    for s in suggestions:
        assert "priority" in s
        assert "action" in s

    print(f"    All pass: {r1.data['pass']}")
    print(f"    Dim fail (light=0.68): pass={r2.data['pass']}, failures={r2.data['threshold_failures']}")
    print(f"    Overall fail: pass={r3.data['pass']}, suggestions={len(suggestions)}")
    print("  P test_quality_aggregator_enhanced")


async def test_benchmark_tool():
    """测试性能基准工具"""
    tool = BenchmarkTool()
    registry = ToolRegistry()
    registry.register(tool)

    # 模拟各阶段耗时
    phases = [
        ("creative", 850, 1), ("character", 1200, 1),
        ("shot_generation", 3200, 5), ("shot_generation", 2800, 5),
        ("shot_generation", 3400, 5), ("lighting", 450, 5),
        ("audio", 2100, 5), ("color_grade", 600, 5),
        ("post", 1800, 1),
    ]
    for phase, ms, batch in phases:
        await registry.call("benchmark", {
            "phase": phase, "elapsed_ms": ms, "shot_count": 1, "batch_size": batch,
        })

    result = await registry.call("benchmark", {"phase": "", "elapsed_ms": 0})
    data = result.data
    phase_stats = data.get("phase_stats", {})

    assert len(phase_stats) >= 5
    assert data["bottleneck"]
    assert data["bottleneck_suggestion"]

    # shot_generation 应该是最大的
    shot_gen = phase_stats.get("shot_generation", {})
    assert shot_gen["calls"] == 3

    print(f"    Phases: {list(phase_stats.keys())}")
    print(f"    Bottleneck: {data['bottleneck']} ({data['bottleneck_suggestion'][:50]}...)")
    print("  P test_benchmark_tool")


# ── 大规模压力测试 ────────────────────────────────────

async def test_stress_scheduler_100_shots():
    """压力测试 — 100 镜头调度"""
    print("\n  -- 100 镜头压力测试 --")

    from src.agents.scheduler import SchedulerAgent
    from src.core.agent import AgentConfig
    from src.core.message_bus import MessageBus

    bus = MessageBus()
    tools = ToolRegistry()
    config = AgentConfig(name="Scheduler", role="scheduler")
    agent = SchedulerAgent(config, bus, tools)
    await agent.start()

    # 生成 100 个镜头: 5 个场景, 3 个角色, 少量依赖
    shots = []
    for i in range(100):
        scene_id = f"scene_{i % 5}"
        shots.append({
            "id": f"s{i:03d}",
            "scene_id": scene_id,
            "characters_in_frame": [f"char_{i % 3}"],
            "dependencies": [f"s{i - 5:03d}"] if i >= 10 and i % 10 == 0 else [],
        })

    start_time = time.time()
    await agent.handle_task({"action": "init", "shotlist": {"shots": shots}, "max_retries": 3})
    init_time = (time.time() - start_time) * 1000

    # 模拟调度循环
    completed = 0
    total_batches = 0
    batch_sizes = []

    while completed < len(shots):
        batch_result = await agent.handle_task({"action": "next_batch"})
        batch = batch_result.get("batch", [])
        if not batch:
            break
        total_batches += 1
        batch_sizes.append(len(batch))
        for sid in batch:
            await agent.handle_task({"action": "mark_done", "shot_id": sid})
            completed += 1

    elapsed = (time.time() - start_time) * 1000

    stats = await agent.handle_task({"action": "stats"})
    s = stats["stats"]

    assert s["completed"] == 100
    assert total_batches > 1

    print(f"    Init: {init_time:.0f}ms, 总共: {elapsed:.0f}ms")
    print(f"    批次数: {total_batches}, avg batch: {s['avg_batch_size']:.1f}, max batch: {s['max_batch_size']}")
    print(f"    GPU 利用率: {s['gpu_utilization']:.0f}%, 吞吐量: {s['throughput_per_min']:.0f} shots/min")
    print(f"    批次分布: avg={sum(batch_sizes)/len(batch_sizes):.1f}, min={min(batch_sizes)}, max={max(batch_sizes)}")
    print("  P test_stress_scheduler_100_shots")


async def test_stress_multi_style():
    """多风格测试 — 50 种风格组合验证"""
    print("\n  -- 50 风格组合测试 --")

    from src.tools.audio_video import ColorGradeTool
    from src.tools.audio_video import VFXSubtitleTool

    tools = ToolRegistry()
    grade_tool = ColorGradeTool()
    vfx_tool = VFXSubtitleTool()
    tools.register(grade_tool)
    tools.register(vfx_tool)

    moods = ["desolate", "warm", "cold", "cinematic", "dark_fantasy", "vibrant", "noir", "sunset", "industrial", "neutral"]
    emotions = ["loneliness", "hope", "tension", "joy", "sadness", "fear", "anger", "serene", "wistful", "neutral"]
    vfx_types = ["film_grain", "vignette_dark", "light_leak", "dust_particles", "subtle_blur",
                 "chromatic_aberration", "lens_flare", "scan_line", "glitch", "none"]

    passed = 0
    failed = 0
    for mood in moods:
        for emotion in emotions[:5]:  # 50 combinations (10 x 5)
            # 调色
            r1 = await tools.call("color_grade", {
                "image_path": f"test_{mood}_{emotion}.png",
                "palette_dominant": mood,
                "mood_descriptor": mood,
                "color_temp_k": 5600,
                "scene_id": f"scene_{mood}_{emotion}",
            })
            # VFX
            vfx = vfx_types[hash(f"{mood}_{emotion}") % len(vfx_types)]
            r2 = await tools.call("vfx_subtitle", {
                "dialog": f"{mood} {emotion} 测试。",
                "shot_id": f"shot_{mood}_{emotion}",
                "duration": 3.0,
                "vfx_type": vfx,
                "scene_mood": emotion,
            })
            if r1.status == "ok" and r2.status == "ok":
                passed += 2
            else:
                failed += 2

    total = passed + failed
    assert total == 100  # 50 × 2
    assert failed == 0

    print(f"    风格组合: {total} calls, 全部通过 ({passed}/{total})")
    print(f"    LUT 覆盖: {len(moods)} 种, VFX 覆盖: {len(vfx_types)} 种")
    print("  P test_stress_multi_style")


async def test_phase5_final_integration():
    """Phase 5 最终集成 — 完整质量闭环"""
    print("\n  -- Phase 5 最终集成测试 --")

    registry = ToolRegistry()
    registry.register(CompositionScorerTool())
    registry.register(RhythmScorerTool())
    registry.register(EmotionAlignmentCheckerTool())
    registry.register(QualityAggregatorTool())
    registry.register(BenchmarkTool())

    # 模拟完整管线的质量评分
    shot_scores = []
    for i in range(20):
        comp = await registry.call("composition_scorer", {
            "image_path": f"frame_{i}.png",
            "shot_spec": {
                "camera": {"framing": "medium"},
                "composition": {"position": "center"},
            },
        })

        emotion = await registry.call("emotion_alignment_checker", {
            "image_path": f"frame_{i}.png",
            "audio_path": f"audio_{i}.wav",
            "target_emotion": ["neutral", "hope", "tension", "joy", "sadness"][i % 5],
            "prev_emotion": "neutral",
            "next_emotion": "neutral",
            "dialog_text": "test dialog",
        })

        quality = await registry.call("quality_aggregator", {
            "composition_score": comp.data["overall"],
            "consistency_score": 0.88,
            "light_score": 0.85,
            "emotion_score": emotion.data["overall"],
            "shot_id": f"shot_{i}",
        })

        shot_scores.append({
            "shot_id": f"shot_{i}",
            "comp": comp.data["overall"],
            "emotion": emotion.data["overall"],
            "quality_overall": quality.data["overall"],
            "passed": quality.data["pass"],
        })

    # 验证质量分布
    passed_count = sum(1 for s in shot_scores if s["passed"])
    avg_quality = sum(s["quality_overall"] for s in shot_scores) / len(shot_scores)
    assert passed_count > 0
    assert avg_quality > 0.7

    # 节奏分析
    timeline = {"clips": [{"duration": 2.5 + (i % 4) * 0.5} for i in range(20)]}
    shots_for_rhythm = [
        {"id": f"shot_{i}", "scene_id": f"scene_{i // 5}", "emotion": "neutral",
         "camera": {"framing": "medium"}} for i in range(20)
    ]
    rhythm = await registry.call("rhythm_scorer", {
        "timeline": timeline,
        "shotlist": {"shots": shots_for_rhythm},
    })
    assert rhythm.data["pacing_match"] > 0

    # Benchmark
    for phase in ["creative", "shot_generation", "audio", "post"]:
        await registry.call("benchmark", {
            "phase": phase, "elapsed_ms": 500 + hash(phase) % 2000,
        })

    bench = await registry.call("benchmark", {})
    assert bench.data["bottleneck"]

    print(f"    20 镜头质量: passed={passed_count}/20, avg={avg_quality:.3f}")
    print(f"    节奏: pacing={rhythm.data['pacing_match']:.3f}, 转场={len(rhythm.data['transition_scores'])}")
    print(f"    Bottleneck: {bench.data['bottleneck']}")
    print("  P test_phase5_final_integration")


async def run_all():
    print("\n" + "=" * 60)
    print("Phase 5 - Quality & Stability Tests")
    print("=" * 60)

    tests = [
        ("RhythmScorer Enhanced", test_rhythm_scorer_enhanced),
        ("EmotionAlignment Enhanced", test_emotion_alignment_enhanced),
        ("QualityAggregator Enhanced", test_quality_aggregator_enhanced),
        ("BenchmarkTool", test_benchmark_tool),
        ("Stress Scheduler 100 Shots", test_stress_scheduler_100_shots),
        ("Stress Multi-Style 50 Combos", test_stress_multi_style),
        ("Phase5 Final Integration", test_phase5_final_integration),
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
