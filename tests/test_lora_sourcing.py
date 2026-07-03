"""LoraSourcing tests — CivitAI search, HuggingFace search, AssetDB registry, fallback chain."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.tools.lora_sourcing import LoraSourcing, LORA_DIR
from src.tools.character_tools import LoRATrainerTool
from src.tools.base import ToolCall, ToolResult


@pytest.fixture
def sourcing():
    return LoraSourcing()


@pytest.fixture
def trainer():
    return LoRATrainerTool()


# ── LoraSourcing init ─────────────────────────────────────────


def test_lora_sourcing_init(sourcing):
    assert sourcing is not None
    assert sourcing._http is None  # lazy init


def test_lora_dir_exists():
    assert LORA_DIR.exists()


# ── Source with mock fallback (offline) ───────────────────────


@pytest.mark.asyncio
async def test_source_falls_back_to_mock_when_offline(sourcing):
    """When offline (no internet), source() should return mock result."""
    result = await sourcing.source(
        character_id="test_char",
        description="a mysterious warrior with glowing eyes",
        trigger_word="test_char_v1",
    )
    assert result["source"] == "mock"
    assert "lora_path" in result
    assert result["trigger_word"] == "test_char_v1"
    # Should create a placeholder file
    path = Path(result["lora_path"])
    assert path.exists()
    assert path.suffix == ".safetensors"


@pytest.mark.asyncio
async def test_source_default_trigger_word(sourcing):
    """When no trigger_word given, should use char_{character_id}."""
    result = await sourcing.source(
        character_id="hero1",
        description="a brave hero",
    )
    assert result["trigger_word"] == "char_hero1"


# ── AssetDB registry ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_source_registers_in_asset_db(sourcing):
    result = await sourcing.source(
        character_id="reg_test",
        description="a robot",
        trigger_word="robot_v1",
    )
    from src.tools.asset_db import asset_db
    record = asset_db.get("char_asset_db", "reg_test:lora")
    assert record is not None
    assert record["data"]["source"] == "mock"
    assert record["data"]["trigger_word"] == "robot_v1"
    assert record["data"]["character_id"] == "reg_test"
    assert asset_db.is_locked("char_asset_db", "reg_test:lora")


# ── LoRATrainerTool integration ──────────────────────────────


@pytest.mark.asyncio
async def test_lora_trainer_fallback_to_mock(trainer):
    """LoRATrainerTool should fall back to mock when sourcing unavailable."""
    call = ToolCall(
        tool="lora_trainer",
        params={
            "character_id": "fallback_test",
            "description": "some character",
            "trigger_word": "fallback_v1",
        },
    )
    result = await trainer.execute(call)
    assert result.status == "ok"
    assert result.data.get("sourced") is False
    assert result.metadata.get("mock") is True


@pytest.mark.asyncio
async def test_lora_trainer_requires_character_id(trainer):
    call = ToolCall(tool="lora_trainer", params={})
    result = await trainer.execute(call)
    assert result.status == "fail"


@pytest.mark.asyncio
async def test_lora_trainer_creates_file(trainer):
    call = ToolCall(
        tool="lora_trainer",
        params={
            "character_id": "file_test",
            "description": "test character",
            "trigger_word": "file_trig",
        },
    )
    result = await trainer.execute(call)
    path = Path(result.data["lora_path"])
    assert path.exists()


# ── Schema ───────────────────────────────────────────────────


def test_lora_sourcing_http_headers(sourcing):
    """Verify HTTP client has proper User-Agent."""
    assert sourcing.http.headers.get("User-Agent", "").startswith("DirectorFactory")


def test_lora_trainer_schema_includes_description(trainer):
    s = trainer.schema()
    params = s["parameters"]
    assert "description" in params
    assert "character_id" in params


# ── CivitAI API structure (offline mock) ─────────────────────


@pytest.mark.asyncio
async def test_try_civitai_returns_none_when_offline(sourcing):
    """When offline, _try_civitai should return None."""
    result = await sourcing._try_civitai("test", "a warrior", "warrior_v1")
    assert result is None


@pytest.mark.asyncio
async def test_try_huggingface_returns_none_when_offline(sourcing):
    """When offline, _try_huggingface should return None."""
    result = await sourcing._try_huggingface("test", "a warrior", "warrior_v1")
    assert result is None


# ── Close ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_close_cleans_up(sourcing):
    await sourcing.close()
    assert sourcing._http is None


@pytest.mark.asyncio
async def test_close_when_no_http():
    s = LoraSourcing()
    await s.close()
    assert s._http is None
