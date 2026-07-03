"""Phase 2 测试 — 角色导演 + 一致性管线"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.base import ToolRegistry
from src.tools.text_gen import TextGenTool
from src.tools.character_tools import (
    LoRATrainerTool,
    EmbedExtractorTool,
    CharacterConsistencyCheckerTool,
    MultiCharacterCompositionTool,
)
from src.tools.asset_db import asset_db
from src.core.agent import AgentConfig
from src.core.message_bus import MessageBus
from src.agents.character_director import CharacterDirectorAgent


async def test_lora_trainer():
    """测试 LoRA 训练工具"""
    tool = LoRATrainerTool()
    registry = ToolRegistry()
    registry.register(tool)

    result = await registry.call("lora_trainer", {
        "character_id": "robot_01",
        "reference_images": ["ref_001.png"],
        "trigger_word": "robot_r7",
        "steps": 800,
    })

    assert result.status == "ok"
    assert result.data.get("lora_path")
    assert result.data.get("trigger_word") == "robot_r7"

    # 验证已存入 asset_db 并锁定
    record = asset_db.get("char_asset_db", "robot_01:lora")
    assert record is not None
    assert record["locked"] is True
    assert record["data"]["trigger_word"] == "robot_r7"

    print(f"    LoRA path: {result.data['lora_path']}")
    print(f"    Trigger: {result.data['trigger_word']}")
    print("  P test_lora_trainer")


async def test_embed_extractor():
    """测试 Embedding 提取工具"""
    tool = EmbedExtractorTool()
    registry = ToolRegistry()
    registry.register(tool)

    result = await registry.call("embed_extractor", {
        "character_id": "robot_01",
        "reference_image": "ref_001.png",
        "method": "arcface",
    })

    assert result.status == "ok"
    embedding = result.data.get("embedding", [])
    assert len(embedding) == 512
    assert all(-0.5 <= v <= 0.5 for v in embedding)

    # 验证已锁定
    record = asset_db.get("char_asset_db", "robot_01:embedding")
    assert record["locked"] is True

    print(f"    Embedding dim: {len(embedding)}")
    print(f"    First 5 values: {embedding[:5]}")
    print("  P test_embed_extractor")


async def test_character_consistency_checker():
    """测试增强一致性检查"""
    tool = CharacterConsistencyCheckerTool()
    registry = ToolRegistry()
    registry.register(tool)

    result = await registry.call("character_consistency_checker", {
        "generated_image_path": "test.png",
        "character_id": "robot_01",
        "reference_embedding": [0.1] * 512,
        "check_dimensions": ["face", "appearance", "style"],
    })

    assert result.status == "ok"
    data = result.data
    assert "face_similarity" in data
    assert "appearance_match" in data
    assert "style_match" in data
    assert "overall" in data
    assert "pass" in data

    print(f"    Face: {data['face_similarity']}, Appearance: {data['appearance_match']}, Style: {data['style_match']}")
    print(f"    Overall: {data['overall']}, Pass: {data['pass']}")
    print("  P test_character_consistency_checker")


async def test_multi_char_composition():
    """测试多角色同框工具"""
    tool = MultiCharacterCompositionTool()
    registry = ToolRegistry()
    registry.register(tool)

    # 单角色
    r1 = await registry.call("multi_char_composition", {
        "generated_image_path": "test.png",
        "character_ids": ["robot_01"],
    })
    assert r1.status == "ok"
    assert r1.data.get("pass") is True

    # 多角色
    r2 = await registry.call("multi_char_composition", {
        "generated_image_path": "test.png",
        "character_ids": ["robot_01", "robot_02", "robot_03"],
    })
    assert r2.status == "ok"
    assert len(r2.data.get("character_results", {})) == 3

    print(f"    Multi-char results: {r2.data['character_results']}")
    print("  P test_multi_char_composition")


async def test_character_director_create():
    """测试角色导演 — 完整创建流程"""
    bus = MessageBus()
    tools = ToolRegistry()
    tools.register(TextGenTool())
    tools.register(LoRATrainerTool())
    tools.register(EmbedExtractorTool())
    tools.register(CharacterConsistencyCheckerTool())
    tools.register(MultiCharacterCompositionTool())

    config = AgentConfig(name="CharacterDirector", role="character_director")
    agent = CharacterDirectorAgent(config, bus, tools)
    await agent.start()

    characters = [
        {
            "id": "robot_01",
            "name": "R-7",
            "description": "锈迹斑斑的人形机器人, left eye sensor cracked, missing right pinky, worn metal texture",
            "voice_style": "机械但温柔, 略带杂音",
        },
        {
            "id": "robot_02",
            "name": "EVE-3",
            "description": "白色光滑外壳的侦察机器人, glowing blue visor, sharp angular design",
            "voice_style": "冰冷, 精确, 不带感情",
        },
    ]

    result = await agent.handle_task({
        "action": "create_all",
        "characters": characters,
        "style_guide": {"visual_mood": "desolate, cold, cinematic"},
    })

    assert result["status"] == "ok"
    profiles = result.get("profiles", {})
    assert len(profiles) == 2

    # 验证每个角色的资产
    for cid, profile in profiles.items():
        assert profile["locked"] is True
        assert profile["lora_path"]
        assert len(profile["face_embedding"]) == 512
        assert len(profile["distinctive_features"]) > 0

        # 验证 asset_db 中有完整资产
        lora = asset_db.get("char_asset_db", f"{cid}:lora")
        embed = asset_db.get("char_asset_db", f"{cid}:embedding")
        prof = asset_db.get("char_asset_db", f"{cid}:profile")
        assert lora is not None
        assert embed is not None
        assert prof is not None
        assert prof["locked"] is True

    print(f"    创建角色: {list(profiles.keys())}")
    print(f"    R-7 features: {profiles['robot_01']['distinctive_features']}")
    print(f"    EVE-3 features: {profiles['robot_02']['distinctive_features']}")
    print("  P test_character_director_create")


async def test_character_consistency_verify():
    """测试角色一致性验证流程"""
    bus = MessageBus()
    tools = ToolRegistry()
    tools.register(CharacterConsistencyCheckerTool())
    tools.register(MultiCharacterCompositionTool())
    tools.register(TextGenTool())
    tools.register(LoRATrainerTool())
    tools.register(EmbedExtractorTool())

    config = AgentConfig(name="CharacterDirector", role="character_director")
    agent = CharacterDirectorAgent(config, bus, tools)
    await agent.start()

    # 先创建一个角色
    await agent.handle_task({
        "action": "create_character",
        "character": {
            "id": "test_char_01",
            "name": "TestBot",
            "description": "A rusty robot with a cracked visor",
            "voice_style": "mechanical",
        },
        "style_guide": {"visual_mood": "cinematic"},
    })

    # 验证单角色一致性
    verify_result = await agent.handle_task({
        "action": "verify",
        "character_id": "test_char_01",
        "image_path": "test_shot.png",
    })

    assert verify_result["status"] == "ok"
    assert "pass" in verify_result
    print(f"    单角色验证: pass={verify_result['pass']}")

    # 验证多角色
    multi_result = await agent.handle_task({
        "action": "verify_multi",
        "character_ids": ["test_char_01", "robot_02"],
        "image_path": "test_shot_multi.png",
    })

    assert multi_result["status"] == "ok"
    print(f"    多角色验证: {multi_result['data']}")
    print("  P test_character_consistency_verify")


async def test_character_pipeline_integration():
    """角色管线集成测试 — 从剧本到锁定角色"""
    print("\n  -- 角色管线集成测试 --")

    bus = MessageBus()
    tools = ToolRegistry()
    tools.register(TextGenTool())
    tools.register(LoRATrainerTool())
    tools.register(EmbedExtractorTool())
    tools.register(CharacterConsistencyCheckerTool())
    tools.register(MultiCharacterCompositionTool())

    config = AgentConfig(name="CharacterDirector", role="character_director")
    agent = CharacterDirectorAgent(config, bus, tools)
    await agent.start()

    # 模拟完整剧本中的角色
    screenplay_chars = [
        {"id": "hero_01", "name": "主角", "description": "伤痕累累的战士, scar on left cheek, tall and muscular", "voice_style": "低沉沙哑"},
        {"id": "side_01", "name": "同伴", "description": "敏捷的侦察兵, short and wiry, sharp eyes", "voice_style": "轻快明亮"},
        {"id": "villain_01", "name": "反派", "description": "巨大的机械生物, glowing red eyes, metallic tentacles", "voice_style": "低沉威胁"},
    ]

    # 创建所有角色
    create_result = await agent.handle_task({
        "action": "create_all",
        "characters": screenplay_chars,
        "style_guide": {"visual_mood": "dark fantasy, cinematic"},
    })

    profiles = create_result.get("profiles", {})
    assert len(profiles) == 3

    # 验证所有角色资产锁定
    for cid in ["hero_01", "side_01", "villain_01"]:
        assert cid in profiles
        assert profiles[cid]["locked"] is True

        # 检查 asset_db 完整性
        lora = asset_db.get("char_asset_db", f"{cid}:lora")
        assert lora is not None and lora["locked"]
        embed = asset_db.get("char_asset_db", f"{cid}:embedding")
        assert embed is not None and embed["locked"]

    # 模拟镜头中的一致性检查
    for cid, profile in profiles.items():
        check = await agent.handle_task({
            "action": "verify",
            "character_id": cid,
            "image_path": f"shot_with_{cid}.png",
        })
        assert check["status"] == "ok"

    # 多角色同框 (hero + side)
    multi = await agent.handle_task({
        "action": "verify_multi",
        "character_ids": ["hero_01", "side_01"],
        "image_path": "scene03_hero_side.png",
    })
    assert multi["status"] == "ok"

    print(f"    角色数: {len(profiles)}")
    for cid, p in profiles.items():
        print(f"      {p['character_id']}: {len(p['face_embedding'])}d embedding, features={p['distinctive_features']}")
    print("  P test_character_pipeline_integration")


async def run_all():
    print("\n" + "=" * 60)
    print("Phase 2 - Character Pipeline Tests")
    print("=" * 60)

    tests = [
        ("LoRATrainer", test_lora_trainer),
        ("EmbedExtractor", test_embed_extractor),
        ("CharConsistencyChecker", test_character_consistency_checker),
        ("MultiCharComposition", test_multi_char_composition),
        ("CharacterDirector Create", test_character_director_create),
        ("CharacterDirector Verify", test_character_consistency_verify),
        ("CharacterPipeline Integration", test_character_pipeline_integration),
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
