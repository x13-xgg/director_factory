"""数据协议定义 — 系统中所有核心数据结构的 Python 实现"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── 枚举 ────────────────────────────────────────────

class Emotion(str, Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    LONELINESS = "loneliness"
    HOPE = "hope"
    TENSION = "tension"
    SERENE = "serene"
    WISTFUL = "wistful"
    NEUTRAL = "neutral"


class TransitionType(str, Enum):
    CUT = "cut"
    DISSOLVE = "dissolve"
    FADE = "fade"
    WIPE = "wipe"
    MATCH = "match"


class Framing(str, Enum):
    EXTREME_WIDE = "extreme_wide"
    WIDE = "wide"
    MEDIUM_WIDE = "medium_wide"
    MEDIUM = "medium"
    MEDIUM_CLOSE = "medium_close"
    CLOSE_UP = "close_up"
    EXTREME_CLOSE_UP = "extreme_close_up"


class CameraAngle(str, Enum):
    EYE_LEVEL = "eye_level"
    LOW = "low"
    HIGH = "high"
    DUTCH = "dutch"
    OVERHEAD = "overhead"


class CameraMovement(str, Enum):
    STATIC = "static"
    PUSH_IN = "push_in"
    PULL_OUT = "pull_out"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    TILT_UP = "tilt_up"
    TILT_DOWN = "tilt_down"
    TRACKING = "tracking"
    HANDHELD = "handheld"


class DepthOfField(str, Enum):
    SHALLOW = "shallow"
    MEDIUM = "medium"
    DEEP = "deep"


class ShotStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    RETRY = "retry"
    FAILED = "failed"


class PipelinePhase(str, Enum):
    CREATIVE = "creative"
    CHARACTER = "character"
    SHOT = "shot"
    POST = "post"


# ── 基础数据类 ─────────────────────────────────────

@dataclass
class CameraSpec:
    framing: Framing = Framing.MEDIUM
    angle: CameraAngle = CameraAngle.EYE_LEVEL
    movement: CameraMovement = CameraMovement.STATIC
    depth_of_field: DepthOfField = DepthOfField.MEDIUM
    custom_notes: str = ""


@dataclass
class Composition:
    subject: str = ""
    position: str = "center"
    background: str = ""
    foreground: str = ""


@dataclass
class Transition:
    type: TransitionType = TransitionType.CUT
    duration: float = 0.0
    overlap: float = 0.0


@dataclass
class LightingHint:
    description: str = ""
    color_temp_k: int = 5600
    intensity: float = 1.0
    key_light_direction: list[float] = field(default_factory=lambda: [0.0, 0.0, -1.0])
    fill_intensity: float = 0.3
    rim_color: str = ""
    volumetrics: str = ""


@dataclass
class AudioHint:
    ambience: str = ""
    foley: str = ""
    bgm_mood: str = ""


# ── 核心协议对象 ───────────────────────────────────

@dataclass
class Character:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    voice_style: str = ""
    appearance_tags: list[str] = field(default_factory=list)


@dataclass
class Scene:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    location: str = ""
    mood: str = ""
    time_of_day: str = "day"
    shots: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class Screenplay:
    """编剧输出: 结构化剧本"""

    title: str = ""
    genre: str = ""
    target_duration: float = 60.0
    characters: list[Character] = field(default_factory=list)
    scenes: list[Scene] = field(default_factory=list)
    emotion_curve: list[float] = field(default_factory=list)
    raw_text: str = ""
    version: int = 1


@dataclass
class Shot:
    """分镜师输出: 单镜头指令"""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    scene_id: str = ""
    shot_number: int = 0
    duration: float = 3.0
    camera: CameraSpec = field(default_factory=CameraSpec)
    composition: Composition = field(default_factory=Composition)
    lighting: LightingHint = field(default_factory=LightingHint)
    audio: AudioHint = field(default_factory=AudioHint)
    dialog: str = ""
    emotion: Emotion = Emotion.NEUTRAL
    emotion_intensity: float = 0.5
    transition_in: Transition = field(default_factory=Transition)
    transition_out: Transition = field(default_factory=Transition)
    dependencies: list[str] = field(default_factory=list)
    characters_in_frame: list[str] = field(default_factory=list)
    action_description: str = ""
    status: ShotStatus = ShotStatus.PENDING


@dataclass
class ShotList:
    """分镜师输出: 完整分镜表"""

    project: str = ""
    total_duration: float = 0.0
    shots: list[Shot] = field(default_factory=list)
    version: int = 1


@dataclass
class VisualSpec:
    """美术指导输出: 场景视觉参数"""

    scene_id: str = ""
    palette_dominant: str = ""
    palette_accent: str = ""
    mood_descriptor: str = ""
    texture_prompt: str = ""
    negative_prompt: str = ""
    reference_style: str = ""
    version: int = 1


@dataclass
class StyleGuide:
    """美术指导输出: 全局风格指南"""

    project: str = ""
    global_palette: str = ""
    lighting_style: str = ""
    visual_mood: str = ""
    visual_specs: dict[str, VisualSpec] = field(default_factory=dict)
    version: int = 1


@dataclass
class CharacterProfile:
    """角色导演输出: 角色资产包"""

    character_id: str = ""
    base_prompt: str = ""
    distinctive_features: list[str] = field(default_factory=list)
    lora_path: str = ""
    face_embedding: list[float] = field(default_factory=list)
    reference_image_path: str = ""
    locked: bool = False
    version: int = 1


@dataclass
class Frame:
    """摄影师输出: 单帧画面"""

    shot_id: str = ""
    image_path: str = ""
    prompt: str = ""
    negative_prompt: str = ""
    seed: int = 0
    composition_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioClip:
    """配音演员输出: 音频片段"""

    shot_id: str = ""
    audio_path: str = ""
    text: str = ""
    emotion: Emotion = Emotion.NEUTRAL
    duration: float = 0.0
    phoneme_timestamps: list[dict] = field(default_factory=list)


@dataclass
class TimelineClip:
    """剪辑师输出: 时间线片段"""

    shot_id: str = ""
    video_path: str = ""
    in_point: float = 0.0
    out_point: float = 0.0
    transition: Transition = field(default_factory=Transition)


@dataclass
class Timeline:
    """剪辑师输出: 完整时间线"""

    project: str = ""
    clips: list[TimelineClip] = field(default_factory=list)
    total_duration: float = 0.0
    fps: int = 24


@dataclass
class ProductionTask:
    """制片调度: 任务单元"""

    shot: Shot
    retry_count: int = 0
    max_retries: int = 3
    feedback: str = ""


@dataclass
class ProductionPlan:
    """总监输出: 生产计划"""

    project_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    scenes: list[dict] = field(default_factory=list)
    style_guide: StyleGuide | None = None
    quality_threshold: float = 0.85
    status: str = "created"


@dataclass
class QualityReport:
    """总监输出: 质量审核报告"""

    shot_id: str = ""
    composition_score: float = 0.0
    consistency_score: float = 0.0
    light_score: float = 0.0
    emotion_score: float = 0.0
    overall: float = 0.0
    passed: bool = False
    feedback: str = ""
    suggestions: list[str] = field(default_factory=list)

    @property
    def weight_map(self) -> dict[str, float]:
        return {
            "composition": 0.40,
            "consistency": 0.35,
            "light": 0.15,
            "emotion": 0.10,
        }
