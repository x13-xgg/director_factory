"""MultiCharacterCompositionTool tests — ArcFace multi-face detection, matching, occlusion, fallback."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from src.tools.character_tools import MultiCharacterCompositionTool
from src.tools.base import ToolCall


@pytest.fixture
def tool():
    return MultiCharacterCompositionTool()


# ── Bbox IoU ──────────────────────────────────────────────────


def test_bbox_iou_no_overlap(tool):
    box_a = [0, 0, 10, 10]
    box_b = [20, 20, 30, 30]
    iou = tool._bbox_iou(box_a, box_b)
    assert iou == 0.0


def test_bbox_iou_full_overlap(tool):
    box = [0, 0, 10, 10]
    iou = tool._bbox_iou(box, box)
    assert iou == 1.0


def test_bbox_iou_partial_overlap(tool):
    box_a = [0, 0, 10, 10]
    box_b = [5, 5, 15, 15]
    iou = tool._bbox_iou(box_a, box_b)
    assert 0.1 < iou < 0.2


def test_bbox_iou_adjacent(tool):
    box_a = [0, 0, 10, 10]
    box_b = [10, 0, 20, 10]
    iou = tool._bbox_iou(box_a, box_b)
    assert iou == 0.0


# ── Single character ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_single_character_skips_check(tool):
    call = ToolCall(
        tool="multi_char_composition",
        params={"character_ids": ["r1"]},
    )
    result = await tool.execute(call)
    assert result.status == "ok"
    assert result.data.get("pass") is True
    assert "Single character" in result.data.get("message", "")


# ── Fallback multi-check ──────────────────────────────────────


@pytest.mark.asyncio
async def test_fallback_multi_check(tool):
    """Without a real image, should use deterministic fallback."""
    call = ToolCall(
        tool="multi_char_composition",
        params={
            "character_ids": ["r1", "r2"],
            "generated_image_path": "/nonexistent/path.png",
        },
    )
    result = await tool.execute(call)
    assert result.status == "ok"
    assert result.data.get("method") == "deterministic_fallback"
    assert "character_results" in result.data
    assert len(result.data["character_results"]) == 2


@pytest.mark.asyncio
async def test_fallback_deterministic_same_input(tool):
    """Same character IDs should produce same fallback results."""
    call1 = ToolCall(
        tool="multi_char_composition",
        params={"character_ids": ["a", "b"]},
    )
    call2 = ToolCall(
        tool="multi_char_composition",
        params={"character_ids": ["a", "b"]},
    )
    r1 = await tool.execute(call1)
    r2 = await tool.execute(call2)
    assert r1.data["character_results"]["a"]["similarity"] == r2.data["character_results"]["a"]["similarity"]


# ── Real ArcFace multi-check ──────────────────────────────────


@pytest.mark.asyncio
async def test_real_multi_check_no_faces(tool, tmp_path):
    """Empty/solid image should detect no faces."""
    from PIL import Image
    img_path = tmp_path / "blank.png"
    img = Image.new("RGB", (512, 512), color=(40, 40, 45))
    img.save(str(img_path))

    call = ToolCall(
        tool="multi_char_composition",
        params={
            "character_ids": ["r1", "r2"],
            "generated_image_path": str(img_path),
        },
    )
    result = await tool.execute(call)
    assert result.status == "ok"
    assert result.data.get("method") == "arcface"
    assert result.data.get("faces_detected") == 0
    assert result.data.get("all_pass") is False
    assert "No faces detected" in result.data.get("suggestions", [""])[0]


@pytest.mark.asyncio
async def test_real_multi_check_with_embeddings_in_db(tool, tmp_path):
    """When reference embeddings exist in AssetDB, they should be used."""
    from src.tools.asset_db import asset_db
    from src.tools.face_utils import deterministic_embedding

    # Store reference embeddings
    for cid in ["char_a", "char_b"]:
        det_emb = deterministic_embedding(cid)
        asset_db.put(
            "char_asset_db",
            f"{cid}:embedding",
            {"character_id": cid, "embedding": [float(v) for v in det_emb]},
            {"type": "embedding"},
        )

    from PIL import Image
    img_path = tmp_path / "test.png"
    img = Image.new("RGB", (512, 512), color=(50, 45, 40))
    img.save(str(img_path))

    call = ToolCall(
        tool="multi_char_composition",
        params={
            "character_ids": ["char_a", "char_b"],
            "generated_image_path": str(img_path),
        },
    )
    result = await tool.execute(call)
    assert result.status == "ok"
    assert result.data.get("method") == "arcface"
    # On blank image, no faces detected
    assert result.data.get("faces_detected") == 0
    assert "char_a" in result.data["character_results"]
    assert "char_b" in result.data["character_results"]


# ── Schema ───────────────────────────────────────────────────


def test_schema_includes_image_path(tool):
    s = tool.schema()
    params = s["parameters"]
    assert "generated_image_path" in params
    assert "character_ids" in params


# ── Thresholds ───────────────────────────────────────────────


def test_thresholds_are_reasonable(tool):
    assert 0 < tool.FACE_SIMILARITY_THRESHOLD < 1.0
    assert 0 < tool.QUALITY_PASS_THRESHOLD < 1.0
    assert tool.QUALITY_PASS_THRESHOLD > tool.FACE_SIMILARITY_THRESHOLD
    assert 0 < tool.OCCLUSION_IOU_THRESHOLD < 1.0


# ── Method reported ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_method_is_arcface_when_image_exists(tool, tmp_path):
    from PIL import Image
    img_path = tmp_path / "exists.png"
    Image.new("RGB", (64, 64)).save(str(img_path))

    call = ToolCall(
        tool="multi_char_composition",
        params={
            "character_ids": ["x", "y"],
            "generated_image_path": str(img_path),
        },
    )
    result = await tool.execute(call)
    assert result.data.get("method") == "arcface"
