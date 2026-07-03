"""TextGenTool boundary tests — _parse_output, _generate_mock_from_schema, DeepSeek detection."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.tools.text_gen import TextGenTool
from src.tools.base import ToolRegistry, ToolCall


@pytest.fixture
def tool():
    return TextGenTool()


# ── _parse_output ──────────────────────────────────────────


def test_parse_output_valid_json(tool):
    result = tool._parse_output('{"key": "value"}', {"type": "object"})
    assert result == {"key": "value"}


def test_parse_output_markdown_json_block(tool):
    content = '```json\n{"name": "test"}\n```'
    result = tool._parse_output(content, {"type": "object"})
    assert result == {"name": "test"}


def test_parse_output_invalid_json(tool):
    result = tool._parse_output("not json at all", {"type": "object"})
    assert result is None


def test_parse_output_empty(tool):
    result = tool._parse_output("", {"type": "object"})
    assert result is None


def test_parse_output_no_schema_returns_raw(tool):
    result = tool._parse_output("plain text", None)
    assert result == "plain text"


# ── _generate_mock_from_schema ─────────────────────────────


def test_mock_simple_object(tool):
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "score": {"type": "number"},
            "active": {"type": "boolean"},
        },
        "required": ["name", "age"],
    }
    result = tool._generate_mock_from_schema(schema)
    assert isinstance(result["name"], str)
    assert isinstance(result["age"], int)
    assert isinstance(result["score"], float)
    assert isinstance(result["active"], bool)
    assert result["name"] != ""
    assert result["age"] == 1  # required
    assert result["active"] is False  # not required


def test_mock_string_enum(tool):
    schema = {
        "type": "object",
        "properties": {
            "mood": {"type": "string", "enum": ["happy", "sad", "angry"]},
        },
    }
    result = tool._generate_mock_from_schema(schema)
    assert result["mood"] == "happy"


def test_mock_array_of_strings(tool):
    schema = {
        "type": "object",
        "properties": {
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["tags"],
    }
    result = tool._generate_mock_from_schema(schema)
    assert isinstance(result["tags"], list)
    assert len(result["tags"]) == 3  # required → 3 items
    assert all(isinstance(t, str) for t in result["tags"])


def test_mock_array_of_enums(tool):
    schema = {
        "type": "object",
        "properties": {
            "genres": {"type": "array", "items": {"type": "string", "enum": ["A", "B", "C", "D"]}},
        },
        "required": ["genres"],
    }
    result = tool._generate_mock_from_schema(schema)
    assert result["genres"] == ["A", "B", "C"]


def test_mock_nested_object(tool):
    schema = {
        "type": "object",
        "properties": {
            "camera": {
                "type": "object",
                "properties": {
                    "framing": {"type": "string"},
                    "angle": {"type": "number"},
                },
            },
        },
    }
    result = tool._generate_mock_from_schema(schema)
    assert isinstance(result["camera"], dict)
    assert "framing" in result["camera"]
    assert "angle" in result["camera"]


def test_mock_array_of_objects(tool):
    schema = {
        "type": "object",
        "properties": {
            "shots": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "duration": {"type": "number"},
                    },
                },
            },
        },
        "required": ["shots"],
    }
    result = tool._generate_mock_from_schema(schema)
    assert isinstance(result["shots"], list)
    assert len(result["shots"]) == 3
    assert "id" in result["shots"][0]


# ── DeepSeek integration (mock mode) ───────────────────────


@pytest.mark.asyncio
async def test_text_gen_with_deepseek_model(tool):
    """Verify text_gen works with deepseek model via mock mode (our fix path)."""
    registry = ToolRegistry()
    registry.register(tool)

    result = await registry.call("text_gen", {
        "model": "deepseek-chat",
        "system_prompt": "You are a helpful assistant.",
        "user_prompt": "Return JSON with keys: title (string), genre (string).",
        "output_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "genre": {"type": "string"},
            },
        },
    })
    assert result.status == "ok"
    assert result.data is not None


@pytest.mark.asyncio
async def test_text_gen_complex_schema(tool):
    """Verify tool handles complex nested schemas."""
    registry = ToolRegistry()
    registry.register(tool)

    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "characters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "traits": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "scenes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "location": {"type": "string"},
                        "mood": {"type": "string", "enum": ["dark", "bright", "neutral"]},
                    },
                },
            },
        },
    }
    result = await registry.call("text_gen", {
        "system_prompt": "You are a screenwriter.",
        "user_prompt": "Create a short film script.",
        "output_schema": schema,
    })
    assert result.status == "ok"
    data = result.data
    assert "title" in data
    assert isinstance(data.get("characters"), list)
    assert isinstance(data.get("scenes"), list)


# ── Tool call construction ─────────────────────────────────


def test_tool_call_creation():
    """Verify ToolCall properly stores params."""
    call = ToolCall(
        tool="text_gen",
        params={
            "system_prompt": "Hello",
            "user_prompt": "World",
            "output_schema": {"type": "object"},
        },
        caller="TestAgent",
    )
    assert call.tool == "text_gen"
    assert call.params["system_prompt"] == "Hello"
    assert call.caller == "TestAgent"
    assert len(call.trace_id) == 16


def test_tool_result_metadata():
    """Verify ToolResult carries metadata correctly."""
    from src.tools.base import ToolResult
    result = ToolResult.ok(data={"key": "val"}, model="test-model", usage={"tokens": 42}, provider="test-provider")
    assert result.status == "ok"
    assert result.data == {"key": "val"}
    assert result.metadata["model"] == "test-model"
    assert result.metadata["provider"] == "test-provider"

    # Test fail status
    fail = ToolResult.fail(data=None, suggestions=["retry with different prompt"])
    assert fail.status == "fail"
    assert fail.suggestions == ["retry with different prompt"]

    # Test retry status
    retry = ToolResult.retry(data=None, suggestions=["wait 10s"])
    assert retry.status == "retry"
    assert retry.suggestions == ["wait 10s"]

    # Test mock metadata
    mock_result = ToolResult.ok(data={"x": 1}, mock=True)
    assert mock_result.metadata["mock"] is True
