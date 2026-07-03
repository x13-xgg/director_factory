"""TranslationTool + language constants tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.tools.translator import (
    TranslationTool,
    get_language_info,
    is_supported_language,
    SUPPORTED_LANGUAGES,
    LANGUAGE_NAMES,
)
from src.tools.base import ToolCall


def test_supported_languages_table():
    for lang in ("zh", "en", "ja", "ko"):
        assert lang in SUPPORTED_LANGUAGES
        info = SUPPORTED_LANGUAGES[lang]
        assert "edge_tts_prefix" in info
        assert "wps" in info
        assert "line_sep" in info


def test_get_language_info_known():
    info = get_language_info("en")
    assert info["wps"] == 3.0
    assert info["line_sep"] == "."


def test_get_language_info_unknown_falls_back():
    info = get_language_info("fr")
    assert info == SUPPORTED_LANGUAGES["zh"]


def test_is_supported():
    assert is_supported_language("zh") is True
    assert is_supported_language("en") is True
    assert is_supported_language("fr") is False


def test_language_names():
    assert LANGUAGE_NAMES["zh"] == "Chinese"
    assert LANGUAGE_NAMES["en"] == "English"
    assert LANGUAGE_NAMES["ja"] == "Japanese"
    assert LANGUAGE_NAMES["ko"] == "Korean"


@pytest.fixture
def tool():
    return TranslationTool()


def test_schema(tool):
    s = tool.schema()
    assert s["name"] == "translate"
    assert "texts" in s["parameters"]
    assert "target_language" in s["parameters"]


@pytest.mark.asyncio
async def test_execute_empty_texts(tool):
    call = ToolCall(tool="translate", params={"texts": [], "target_language": "en"})
    result = await tool.execute(call)
    assert result.status == "ok"
    assert result.data == {"translations": {}}


@pytest.mark.asyncio
async def test_execute_unsupported_language(tool):
    call = ToolCall(tool="translate", params={"texts": ["hello"], "target_language": "fr"})
    result = await tool.execute(call)
    assert result.status == "fail"


@pytest.mark.asyncio
async def test_execute_identity_same_language(tool):
    call = ToolCall(tool="translate", params={
        "texts": ["你好", "世界"],
        "target_language": "zh",
        "source_language": "zh",
    })
    result = await tool.execute(call)
    assert result.status == "ok"
    assert result.data["translations"] == {"你好": "你好", "世界": "世界"}
    assert result.data["method"] == "identity"


@pytest.mark.asyncio
async def test_execute_fallback_on_api_error(tool):
    """Translation produces a result with either 'llm' or 'mock' method."""
    call = ToolCall(tool="translate", params={
        "texts": ["hello world"],
        "target_language": "en",
        "source_language": "zh",
    })
    result = await tool.execute(call)
    assert result.status == "ok"
    assert "translations" in result.data
    # method is "identity" | "llm" | "mock"
    assert result.data["method"] in ("identity", "llm", "mock")
