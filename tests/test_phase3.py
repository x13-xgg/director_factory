"""Phase 3 测试 — 灯光师 + 配音演员 + 音效师 + 调色师 + 字幕/VFX"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.base import ToolRegistry
from src.tools.text_gen import TextGenTool
from src.tools.audio_video import (
    TTSTool,
    SFXMatcherTool,
    BGMMatcherTool,
    ColorGradeTool,
    VFXSubtitleTool,
)
from src.tools.asset_db import asset_db
from src.core.agent import AgentConfig
from src.core.message_bus import MessageBus
from src.agents.lighting_td import LightingTDAgent
from src.agents.voice_actor import VoiceActorAgent
from src.agents.sound_designer import SoundDesignerAgent
from src.agents.colorist import ColoristAgent
from src.agents.vfx_subtitles import VFXSubtitlesAgent


# ── 工具级测试 ────────────────────────────────────────

async def test_tts_tool():
    """测试 TTS 工具"""
    tool = TTSTool()
    registry = ToolRegistry()
    registry.register(tool)

    result = await registry.call("tts", {
        "text": "你找到它了吗？",
        "emotion": "hope",
        "character_id": "robot_01",
        "voice_profile": "robot_warm",
        "speed": 0.92,
        "pitch_variance": 0.15,
    })

    assert result.status == "ok"
    data = result.data
    assert data.get("audio_path")
    assert data.get("duration", 0) > 0
    assert data.get("emotion") == "hope"

    # 真实 TTS 后端不生成 phoneme_timestamps
    phonemes = data.get("phoneme_timestamps", [])
    gen_method = data.get("gen_method", "mock")
    if gen_method == "mock":
        assert len(phonemes) > 0

    print(f"    TTS duration: {data['duration']:.2f}s, method={gen_method}, phonemes: {len(phonemes)}")
    print("  P test_tts_tool")


async def test_sfx_matcher_tool():
    """测试 SFX 匹配工具"""
    tool = SFXMatcherTool()
    registry = ToolRegistry()
    registry.register(tool)

    result = await registry.call("sfx_matcher", {
        "action_description": "机器人在废墟中行走，踩过碎玻璃",
        "location": "废弃工厂",
        "mood": "tension",
        "max_results": 3,
    })

    assert result.status == "ok"
    data = result.data
    assert len(data.get("sfx_matches", [])) > 0
    print(f"    SFX matches: {[m['keyword'] for m in data['sfx_matches']]}")
    print("  P test_sfx_matcher_tool")


async def test_bgm_matcher_tool():
    """测试 BGM 匹配工具"""
    tool = BGMMatcherTool()
    registry = ToolRegistry()
    registry.register(tool)

    result = await registry.call("bgm_matcher", {
        "emotion": "hope",
        "scene_mood": "cinematic",
        "intensity": 0.7,
    })

    assert result.status == "ok"
    data = result.data
    assert data.get("bgm_path")
    assert data.get("bpm", 0) > 0
    print(f"    BGM: {data['style']} ({data['bpm']} bpm, {data['key']})")
    print("  P test_bgm_matcher_tool")


async def test_color_grade_tool():
    """测试调色工具"""
    tool = ColorGradeTool()
    registry = ToolRegistry()
    registry.register(tool)

    result = await registry.call("color_grade", {
        "image_path": "test_frame.png",
        "palette_dominant": "cold blue",
        "palette_accent": "warm gold",
        "mood_descriptor": "desolate",
        "color_temp_k": 4500,
        "scene_id": "scene_01",
    })

    assert result.status == "ok"
    data = result.data
    assert data.get("lut_applied")
    grade = data.get("grade_params", {})
    assert "exposure" in grade
    assert "contrast" in grade
    print(f"    LUT: {data['lut_applied']}, temp={grade['temperature']}")
    print("  P test_color_grade_tool")


async def test_vfx_subtitle_tool():
    """测试字幕/VFX 工具"""
    tool = VFXSubtitleTool()
    registry = ToolRegistry()
    registry.register(tool)

    result = await registry.call("vfx_subtitle", {
        "dialog": "你找到它了吗？；找到了，它就在这里。",
        "shot_id": "shot_03",
        "duration": 4.0,
        "vfx_type": "dust_particles",
        "scene_mood": "loneliness",
    })

    assert result.status == "ok"
    data = result.data
    assert data.get("subtitle_count", 0) == 2
    assert "你找到它了吗" in data.get("srt_content", "")
    assert data.get("auto_vfx_suggestion") == "dust_particles"
    print(f"    Subtitles: {data['subtitle_count']}, VFX auto: {data['auto_vfx_suggestion']}")
    print("  P test_vfx_subtitle_tool")


# ── Agent 级测试 ───────────────────────────────────────

async def test_lighting_td_agent():
    """测试灯光师 Agent"""
    bus = MessageBus()
    tools = ToolRegistry()
    tools.register(TextGenTool())

    config = AgentConfig(name="LightingTD", role="lighting_td")
    agent = LightingTDAgent(config, bus, tools)
    await agent.start()

    shot = {
        "id": "shot_01",
        "scene_id": "scene_ruins",
        "emotion": "loneliness",
        "framing": "wide",
        "camera_movement": "static",
        "dialog": False,
        "lighting_description": "废墟中微弱的光线",
        "color_temp": 4500,
    }

    result = await agent.handle_task({
        "action": "generate",
        "shot": shot,
        "style_guide": {"visual_specs": {}},
    })

    assert result["status"] == "ok"
    lighting = result.get("lighting", {})
    assert "key_light" in lighting
    assert "fill_light" in lighting
    assert "rim_light" in lighting
    assert lighting["key_light"]["direction"] == [0.0, 0.3, -0.9]  # loneliness direction
    assert abs(lighting["fill_light"]["intensity"] - 0.18) < 0.02  # 0.15 * 1.2 (wide)

    # 场景缓存
    assert "scene_ruins" in agent._scene_light_cache

    # 连续性检查
    cont_result = await agent.handle_task({
        "action": "check_continuity",
        "prev_histogram": [0.5] * 10,
        "current_histogram": [0.6] * 10,
        "scene_id": "scene_ruins",
    })
    assert cont_result["status"] == "ok"

    print(f"    Key direction: {lighting['key_light']['direction']}, fill: {lighting['fill_light']['intensity']:.2f}")
    print("  P test_lighting_td_agent")


async def test_voice_actor_agent():
    """测试配音演员 Agent"""
    bus = MessageBus()
    tools = ToolRegistry()
    tools.register(TTSTool())

    config = AgentConfig(name="VoiceActor", role="voice_actor")
    agent = VoiceActorAgent(config, bus, tools)
    await agent.start()

    shot = {
        "id": "shot_01",
        "dialog": "你找到它了吗？",
        "emotion": "hope",
        "characters_in_frame": ["robot_01"],
    }

    char_profiles = {
        "robot_01": {"voice_style": "机械但温柔, 略带杂音"},
    }

    result = await agent.handle_task({
        "action": "generate",
        "shot": shot,
        "char_profiles": char_profiles,
    })

    assert result["status"] == "ok"
    clip = result.get("audio_clip", {})
    assert clip.get("audio_path")
    assert clip.get("duration", 0) > 0
    assert clip.get("emotion") == "hope"

    # 批量测试
    shots = [
        {"id": "shot_02", "dialog": "找到了。", "emotion": "joy", "characters_in_frame": ["robot_01"]},
        {"id": "shot_03", "dialog": "", "emotion": "neutral", "characters_in_frame": []},
    ]
    batch_result = await agent.handle_task({
        "action": "generate_all",
        "shots": shots,
        "char_profiles": char_profiles,
    })
    assert batch_result["status"] == "ok"
    assert batch_result["count"] == 1  # only shot_02 has dialog

    print(f"    Clip: {clip['audio_path']}, duration={clip['duration']:.2f}s")
    print("  P test_voice_actor_agent")


async def test_sound_designer_agent():
    """测试音效师 Agent"""
    bus = MessageBus()
    tools = ToolRegistry()
    tools.register(SFXMatcherTool())
    tools.register(BGMMatcherTool())

    config = AgentConfig(name="SoundDesigner", role="sound_designer")
    agent = SoundDesignerAgent(config, bus, tools)
    await agent.start()

    shot = {
        "id": "shot_01",
        "scene_id": "scene_ruins",
        "action_description": "机器人踩过废墟中的玻璃碎片",
        "emotion": "tension",
        "lighting": {"description": "废墟"},
    }

    # 音效设计
    design_result = await agent.handle_task({
        "action": "design",
        "shot": shot,
        "style_guide": {"visual_mood": "desolate"},
    })
    assert design_result["status"] == "ok"
    hints = design_result.get("audio_hints", {})
    assert "foley" in hints
    assert "ambience" in hints

    # BGM 选择
    bgm_result = await agent.handle_task({
        "action": "select_bgm",
        "emotion": "tension",
        "scene_id": "scene_ruins",
    })
    assert bgm_result["status"] == "ok"
    assert bgm_result.get("bgm_path")

    # 场景环境音
    scene_result = await agent.handle_task({
        "action": "design_scene",
        "scene_id": "scene_factory",
        "location": "废弃工厂",
        "mood": "tension",
    })
    assert scene_result["status"] == "ok"

    print(f"    Foley: {len(hints['foley'])}, Ambience: {hints['ambience']}")
    print(f"    BGM: {bgm_result['bgm_info']['style']} ({bgm_result['bgm_info']['bpm']} bpm)")
    print("  P test_sound_designer_agent")


async def test_colorist_agent():
    """测试调色师 Agent"""
    bus = MessageBus()
    tools = ToolRegistry()
    tools.register(ColorGradeTool())

    config = AgentConfig(name="Colorist", role="colorist")
    agent = ColoristAgent(config, bus, tools)
    await agent.start()

    shot = {
        "id": "shot_01",
        "scene_id": "scene_01",
        "emotion": "loneliness",
        "camera_movement": "static",
        "lighting": {"color_temp_k": 4500},
    }

    style_guide = {
        "visual_specs": {
            "scene_01": {
                "palette_dominant": "cool steel blue",
                "palette_accent": "warm rust orange",
                "mood_descriptor": "desolate",
            },
        },
        "visual_mood": "desolate, cold, cinematic",
    }

    # 单镜头调色
    result = await agent.handle_task({
        "action": "grade_shot",
        "shot": shot,
        "style_guide": style_guide,
    })
    assert result["status"] == "ok"
    assert result.get("lut_applied")
    grade = result.get("grade_params", {})
    assert "exposure" in grade

    # 批量
    shots = [
        {"id": "shot_01", "scene_id": "scene_01", "emotion": "loneliness", "camera_movement": "static", "lighting": {"color_temp_k": 4500}},
        {"id": "shot_02", "scene_id": "scene_01", "emotion": "hope", "camera_movement": "handheld", "lighting": {"color_temp_k": 4500}},
    ]
    batch_result = await agent.handle_task({
        "action": "grade_all",
        "shots": shots,
        "style_guide": style_guide,
    })
    assert batch_result["status"] == "ok"
    assert batch_result["count"] == 2

    # 色彩连续性
    cont_result = await agent.handle_task({
        "action": "check_continuity",
        "shots": shots,
    })
    assert cont_result["status"] == "ok"

    print(f"    LUT: {result['lut_applied']}, temperature: {grade.get('temperature', 0):.2f}")
    print("  P test_colorist_agent")


async def test_vfx_subtitles_agent():
    """测试字幕/VFX Agent"""
    bus = MessageBus()
    tools = ToolRegistry()
    tools.register(VFXSubtitleTool())

    config = AgentConfig(name="VFXSubtitles", role="vfx_subtitles")
    agent = VFXSubtitlesAgent(config, bus, tools)
    await agent.start()

    shot = {
        "id": "shot_01",
        "dialog": "你找到它了吗？",
        "duration": 3.0,
        "emotion": "loneliness",
    }

    # 字幕生成
    sub_result = await agent.handle_task({
        "action": "generate_subtitles",
        "shot": shot,
    })
    assert sub_result["status"] == "ok"
    subtitle = sub_result.get("subtitle", {})
    assert subtitle.get("subtitle_count", 0) > 0

    # VFX 应用
    vfx_result = await agent.handle_task({
        "action": "apply_vfx",
        "shot": shot,
        "vfx_type": "dust_particles",
    })
    assert vfx_result["status"] == "ok"
    vfx = vfx_result.get("vfx", {})
    assert vfx.get("vfx_type") == "dust_particles"

    # 一站式
    proc_result = await agent.handle_task({
        "action": "process_shot",
        "shot": shot,
    })
    assert proc_result["status"] == "ok"
    assert proc_result.get("subtitle")
    assert proc_result.get("vfx")

    # SRT 导出
    shots = [
        {"id": "shot_01", "dialog": "你找到它了吗？", "duration": 3.0, "emotion": "loneliness"},
        {"id": "shot_02", "dialog": "找到了。", "duration": 2.0, "emotion": "hope"},
    ]
    await agent.handle_task({"action": "generate_all_subtitles", "shots": shots})
    export_result = await agent.handle_task({
        "action": "export_srt",
        "shots": shots,
        "output_path": "outputs/test_subtitles.srt",
    })
    assert export_result["status"] == "ok"
    assert export_result.get("subtitle_count", 0) == 2

    print(f"    Subtitle lines: {subtitle['subtitle_count']}, VFX type: {vfx['vfx_type']}")
    print("  P test_vfx_subtitles_agent")


async def test_phase3_integration():
    """Phase 3 集成测试 — 所有新 Agent 协作"""
    print("\n  -- Phase 3 集成测试 --")

    bus = MessageBus()
    tools = ToolRegistry()
    tools.register(TextGenTool())
    tools.register(TTSTool())
    tools.register(SFXMatcherTool())
    tools.register(BGMMatcherTool())
    tools.register(ColorGradeTool())
    tools.register(VFXSubtitleTool())

    # 创建所有 Phase 3 Agent
    agents = {
        "LightingTD": LightingTDAgent(AgentConfig(name="LightingTD", role="lighting_td"), bus, tools),
        "VoiceActor": VoiceActorAgent(AgentConfig(name="VoiceActor", role="voice_actor"), bus, tools),
        "SoundDesigner": SoundDesignerAgent(AgentConfig(name="SoundDesigner", role="sound_designer"), bus, tools),
        "Colorist": ColoristAgent(AgentConfig(name="Colorist", role="colorist"), bus, tools),
        "VFXSubtitles": VFXSubtitlesAgent(AgentConfig(name="VFXSubtitles", role="vfx_subtitles"), bus, tools),
    }

    for agent in agents.values():
        await agent.start()

    # 模拟完整镜头
    shots = [
        {
            "id": "shot_01", "scene_id": "ruins_01", "emotion": "loneliness",
            "framing": "wide", "camera_movement": "push_in", "dialog": "这里什么都没有...",
            "characters_in_frame": ["robot_01"], "duration": 4.0,
            "action_description": "机器人独自走在废墟中", "lighting": {"description": "废墟"},
        },
        {
            "id": "shot_02", "scene_id": "ruins_01", "emotion": "hope",
            "framing": "close_up", "camera_movement": "static", "dialog": "等等...那是什么？",
            "characters_in_frame": ["robot_01"], "duration": 3.5,
            "action_description": "机器人发现了什么，蹲下查看", "lighting": {"description": "废墟"},
        },
        {
            "id": "shot_03", "scene_id": "ruins_02", "emotion": "joy",
            "framing": "extreme_close_up", "camera_movement": "static", "dialog": "一朵花。",
            "characters_in_frame": ["robot_01"], "duration": 5.0,
            "action_description": "废墟裂缝中有一朵小花", "lighting": {"description": "裂缝中的阳光"},
        },
    ]

    style_guide = {
        "visual_specs": {
            "ruins_01": {"palette_dominant": "cold gray", "palette_accent": "rust", "mood_descriptor": "desolate"},
            "ruins_02": {"palette_dominant": "warm gold", "palette_accent": "green", "mood_descriptor": "hope"},
        },
        "visual_mood": "desolate, cinematic",
    }

    char_profiles = {
        "robot_01": {"voice_style": "机械但温柔, 略带杂音"},
    }

    # 1. LightingTD — 为每个镜头生成光照
    lightings = {}
    for shot in shots:
        result = await agents["LightingTD"].handle_task({
            "action": "generate", "shot": shot, "style_guide": style_guide,
        })
        lightings[shot["id"]] = result.get("lighting", {})

    # 场景缓存一致性
    assert "ruins_01" in agents["LightingTD"]._scene_light_cache

    # 2. VoiceActor — 生成对白
    voice_result = await agents["VoiceActor"].handle_task({
        "action": "generate_all", "shots": shots, "char_profiles": char_profiles,
    })
    dialog_clips = voice_result.get("clips", {})
    assert len(dialog_clips) == 3

    # 3. SoundDesigner — 音效设计 + BGM
    sfx_designs = {}
    for shot in shots:
        design = await agents["SoundDesigner"].handle_task({
            "action": "design", "shot": shot, "style_guide": style_guide,
        })
        sfx_designs[shot["id"]] = design.get("audio_hints", {})

    bgm_result = await agents["SoundDesigner"].handle_task({
        "action": "select_bgm", "emotion": "hope",
    })
    assert bgm_result.get("bgm_path")

    # 4. Colorist — 调色
    grade_result = await agents["Colorist"].handle_task({
        "action": "grade_all", "shots": shots, "style_guide": style_guide,
    })
    assert grade_result["count"] == 3

    # 5. VFXSubtitles — 字幕 + VFX
    sub_result = await agents["VFXSubtitles"].handle_task({
        "action": "generate_all_subtitles", "shots": shots,
    })
    assert sub_result["count"] == 3

    srt_result = await agents["VFXSubtitles"].handle_task({
        "action": "export_srt", "shots": shots, "output_path": "outputs/test_integration.srt",
    })
    assert srt_result.get("subtitle_count") == 3

    print(f"    光照场景数: {len(agents['LightingTD']._scene_light_cache)}")
    print(f"    对白片段数: {len(dialog_clips)}")
    print(f"    音效设计数: {len(sfx_designs)}")
    print(f"    调色镜头数: {grade_result['count']}")
    print(f"    字幕数: {sub_result['count']}")
    print("  P test_phase3_integration")


async def run_all():
    print("\n" + "=" * 60)
    print("Phase 3 - Dynamic & Audio Pipeline Tests")
    print("=" * 60)

    tests = [
        ("TTSTool", test_tts_tool),
        ("SFXMatcherTool", test_sfx_matcher_tool),
        ("BGMMatcherTool", test_bgm_matcher_tool),
        ("ColorGradeTool", test_color_grade_tool),
        ("VFXSubtitleTool", test_vfx_subtitle_tool),
        ("LightingTDAgent", test_lighting_td_agent),
        ("VoiceActorAgent", test_voice_actor_agent),
        ("SoundDesignerAgent", test_sound_designer_agent),
        ("ColoristAgent", test_colorist_agent),
        ("VFXSubtitlesAgent", test_vfx_subtitles_agent),
        ("Phase3 Integration", test_phase3_integration),
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
