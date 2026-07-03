"""CloudImageGenTool tests — provider chain, fallback, LoRA workflow, mock generation."""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.tools.cloud_image_gen import CloudImageGenTool
from src.tools.base import ToolCall


@pytest.fixture
def tool():
    return CloudImageGenTool()


# ── Provider chain ──────────────────────────────────────────


def test_build_provider_chain_default(tool):
    chain = tool._build_provider_chain()
    assert "mock" in chain


def test_build_provider_chain_no_cloud_keys(tool):
    """Without RunPod/Replicate/Modal keys, only comfyui+mock remain."""
    chain = tool._build_provider_chain()
    # comfyui should be first (no keys needed), mock last
    assert chain[0] == "comfyui"
    assert chain[-1] == "mock"
    assert "runpod" not in chain
    assert "replicate" not in chain
    assert "modal" not in chain


def test_build_provider_chain_includes_configured(tool, monkeypatch):
    """When keys are set, providers appear in chain."""
    monkeypatch.setattr("src.core.config.config.cloud_image_gen.runpod_api_key", "test-key")
    monkeypatch.setattr("src.core.config.config.cloud_image_gen.runpod_endpoint_id", "test-id")
    monkeypatch.setattr("src.core.config.config.cloud_image_gen.provider_order", ["comfyui", "runpod", "mock"])
    chain = tool._build_provider_chain()
    assert "runpod" in chain


def test_provider_order_respected(tool, monkeypatch):
    monkeypatch.setattr("src.core.config.config.cloud_image_gen.runpod_api_key", "test-key")
    monkeypatch.setattr("src.core.config.config.cloud_image_gen.runpod_endpoint_id", "test-id")
    monkeypatch.setattr("src.core.config.config.cloud_image_gen.provider_order", ["runpod", "comfyui", "mock"])
    chain = tool._build_provider_chain()
    assert chain[0] == "runpod"
    assert chain[-1] == "mock"


# ── Mock generation ────────────────────────────────────────


def test_mock_generate_creates_file(tool, tmp_path):
    result = tool._mock_generate(
        "a test prompt", "bad quality",
        64, 64, tmp_path, "test.png", 42,
    )
    assert result.status == "ok"
    images = result.data.get("images", [])
    assert len(images) == 1
    assert Path(images[0]).exists()
    assert result.data.get("gen_method") == "mock"
    assert result.data.get("seed") == 42


def test_mock_generate_default_seed(tool, tmp_path):
    result = tool._mock_generate(
        "prompt", "", 32, 32, tmp_path, "", -1,
    )
    assert result.data.get("seed") != -1


# ── LoRA-aware workflow ─────────────────────────────────────


class TestLoraWorkflow:
    def test_workflow_without_lora(self, tool):
        wf = tool._build_comfyui_workflow(
            "prompt", "neg", 1024, 576, 4, 2.0, 42, 1,
        )
        # 7 nodes: CheckpointLoader, CLIPTextEncode x2, EmptyLatent, KSampler, VAEDecode, SaveImage
        assert len(wf) == 7
        assert wf["1"]["class_type"] == "CheckpointLoaderSimple"
        assert "LoraLoader" not in [n["class_type"] for n in wf.values()]

    def test_workflow_with_lora(self, tool):
        wf = tool._build_comfyui_workflow(
            "prompt", "neg", 1024, 576, 4, 2.0, 42, 1,
            lora_path="assets/loras/test_char.safetensors",
            trigger_word="test_char_v1",
        )
        # 8 nodes: + LoraLoader
        assert len(wf) == 8
        class_types = [n["class_type"] for n in wf.values()]
        assert "LoraLoader" in class_types

    def test_workflow_trigger_word_prepended(self, tool):
        wf = tool._build_comfyui_workflow(
            "a beautiful character", "neg", 1024, 576, 4, 2.0, 42, 1,
            trigger_word="zhangsan_v1",
        )
        clip_node = wf.get("2", {})
        prompt_text = clip_node.get("inputs", {}).get("text", "")
        assert "zhangsan_v1" in prompt_text
        assert prompt_text.startswith("zhangsan_v1")

    def test_workflow_trigger_word_not_duplicated(self, tool):
        """When trigger word already in prompt, don't duplicate."""
        wf = tool._build_comfyui_workflow(
            "zhangsan_v1, a beautiful character", "neg", 1024, 576, 4, 2.0, 42, 1,
            trigger_word="zhangsan_v1",
        )
        clip_node = wf.get("2", {})
        prompt_text = clip_node.get("inputs", {}).get("text", "")
        # should only appear once
        assert prompt_text.count("zhangsan_v1") == 1

    def test_workflow_lora_node_connections(self, tool):
        wf = tool._build_comfyui_workflow(
            "prompt", "neg", 1024, 576, 4, 2.0, 42, 1,
            lora_path="assets/loras/char.safetensors",
            trigger_word="char_v1",
        )
        # With lora, nodes 4=LoraLoader, 5=EmptyLatent, 6=KSampler, 7=VAEDecode, 8=SaveImage
        lora_node = wf.get("4", {})
        assert lora_node["class_type"] == "LoraLoader"
        assert lora_node["inputs"]["model"] == ["1", 0]
        assert lora_node["inputs"]["clip"] == ["1", 1]

        ksampler = wf.get("6", {})
        # model should come from LoraLoader
        assert ksampler["inputs"]["model"] == ["4", 0]
        # positive/negative should come from CLIP encoders
        assert ksampler["inputs"]["positive"] == ["2", 0]
        assert ksampler["inputs"]["negative"] == ["3", 0]


# ── Schema ──────────────────────────────────────────────────


def test_schema_includes_lora_params(tool):
    s = tool.schema()
    params = s["parameters"]
    assert "lora_path" in params
    assert "trigger_word" in params
    assert "prompt" in params


# ── Execute with mock fallback ──────────────────────────────


@pytest.mark.asyncio
async def test_execute_falls_back_to_mock(tool, tmp_path, monkeypatch):
    """When ComfyUI is unreachable, should fall back to mock."""
    monkeypatch.setattr("src.core.config.config.cloud_image_gen.provider_order", ["comfyui", "mock"])
    monkeypatch.setattr("src.core.config.config.cloud_image_gen.comfyui_url", "http://127.0.0.1:19999")  # nothing here

    call = ToolCall(
        tool="cloud_image_gen",
        caller="test",
        params={
            "prompt": "a cat",
            "width": 64,
            "height": 64,
            "output_dir": str(tmp_path),
            "filename": "cat.png",
        },
    )
    result = await tool.execute(call)
    assert result.status == "ok"
    assert result.data.get("gen_method") == "mock"
    assert len(result.data.get("images", [])) == 1
    assert Path(result.data["images"][0]).exists()


@pytest.mark.asyncio
async def test_execute_requires_prompt(tool):
    call = ToolCall(tool="cloud_image_gen", params={})
    result = await tool.execute(call)
    assert result.status == "fail"


# ── Config integration ──────────────────────────────────────


def test_config_provider_order_reads_env(monkeypatch):
    monkeypatch.setenv("CLOUD_IMG_PROVIDER_ORDER", "replicate,runpod,mock")
    from src.core.config import CloudImageGenConfig
    cfg = CloudImageGenConfig()
    assert cfg.provider_order == ["replicate", "runpod", "mock"]


def test_config_runpod_keys_read_env(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rpk-test")
    monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "ep-123")
    from src.core.config import CloudImageGenConfig
    cfg = CloudImageGenConfig()
    assert cfg.runpod_api_key == "rpk-test"
    assert cfg.runpod_endpoint_id == "ep-123"


# ── ToolCall params ─────────────────────────────────────────


def test_toolcall_default_params(tool):
    call = ToolCall(tool="cloud_image_gen", params={"prompt": "test"})
    params = call.params
    assert params.get("prompt") == "test"
