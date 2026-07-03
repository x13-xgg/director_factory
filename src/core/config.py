"""生产配置系统 — 环境变量、API 密钥、后端选择、部署参数"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# 自动加载 .env (项目根目录)
load_dotenv(Path(__file__).parent.parent.parent / ".env")
load_dotenv()  # 也尝试当前目录


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, ""))
    except (ValueError, TypeError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, ""))
    except (ValueError, TypeError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key, "").lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default


@dataclass
class LLMConfig:
    """LLM API 配置"""
    provider: Literal["anthropic", "openai", "deepseek", "auto"] = "auto"
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    anthropic_base_url: str = field(default_factory=lambda: _env("ANTHROPIC_BASE_URL", "https://api.anthropic.com"))
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    openai_base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    default_model: str = field(default_factory=lambda: _env("LLM_DEFAULT_MODEL", "claude-sonnet-4-6"))
    fallback_model: str = field(default_factory=lambda: _env("LLM_FALLBACK_MODEL", "gpt-4o"))
    max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 4096))
    temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.7))
    request_timeout: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT", 120.0))
    max_retries: int = field(default_factory=lambda: _env_int("LLM_MAX_RETRIES", 3))
    retry_backoff: float = field(default_factory=lambda: _env_float("LLM_RETRY_BACKOFF", 2.0))


@dataclass
class ImageGenConfig:
    """图像生成 API 配置"""
    provider: Literal["comfyui", "sdxl", "automatic1111", "mock"] = field(
        default_factory=lambda: _env("IMG_PROVIDER", "mock")
    )
    comfyui_url: str = field(default_factory=lambda: _env("COMFYUI_URL", "http://localhost:8188"))
    comfyui_ws_url: str = field(default_factory=lambda: _env("COMFYUI_WS_URL", "ws://localhost:8188"))
    sdxl_model: str = field(default_factory=lambda: _env("SDXL_MODEL", "DreamShaper XL v2.1 Turbo 闪电_v2.1 Turbo.safetensors"))
    lora_dir: str = field(default_factory=lambda: _env("LORA_DIR", "assets/loras"))
    default_steps: int = field(default_factory=lambda: _env_int("IMG_STEPS", 30))
    default_cfg: float = field(default_factory=lambda: _env_float("IMG_CFG", 7.5))
    default_width: int = field(default_factory=lambda: _env_int("IMG_WIDTH", 1024))
    default_height: int = field(default_factory=lambda: _env_int("IMG_HEIGHT", 576))
    batch_size: int = field(default_factory=lambda: _env_int("IMG_BATCH_SIZE", 1))


@dataclass
class CloudImageGenConfig:
    """Cloud GPU 图像生成 — 多 Provider 回退链"""
    provider_order: list[str] = field(
        default_factory=lambda: [p.strip() for p in _env("CLOUD_IMG_PROVIDER_ORDER", "comfyui,mock").split(",")]
    )
    comfyui_url: str = field(default_factory=lambda: _env("COMFYUI_URL", "http://127.0.0.1:8188"))
    comfyui_ws_url: str = field(default_factory=lambda: _env("COMFYUI_WS_URL", "ws://127.0.0.1:8188"))
    runpod_api_key: str = field(default_factory=lambda: _env("RUNPOD_API_KEY"))
    runpod_endpoint_id: str = field(default_factory=lambda: _env("RUNPOD_ENDPOINT_ID"))
    replicate_api_key: str = field(default_factory=lambda: _env("REPLICATE_API_KEY"))
    replicate_model: str = field(default_factory=lambda: _env("REPLICATE_MODEL", "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"))
    modal_endpoint: str = field(default_factory=lambda: _env("MODAL_ENDPOINT"))
    modal_api_key: str = field(default_factory=lambda: _env("MODAL_API_KEY"))


@dataclass
class TTSConfig:
    """TTS 语音合成配置"""
    provider: Literal["bark", "chattts", "edge_tts", "mock"] = field(
        default_factory=lambda: _env("TTS_PROVIDER", "edge_tts")
    )
    bark_url: str = field(default_factory=lambda: _env("BARK_URL", "http://localhost:5000"))
    chattts_model_path: str = field(default_factory=lambda: _env("CHATTS_MODEL_PATH", "assets/models/chattts"))
    tts_sample_rate: int = field(default_factory=lambda: _env_int("TTS_SAMPLE_RATE", 24000))
    tts_language: str = field(default_factory=lambda: _env("TTS_LANGUAGE", "zh"))


@dataclass
class MessageBusConfig:
    """消息总线后端配置"""
    backend: Literal["memory", "redis", "nats"] = field(
        default_factory=lambda: _env("MSG_BACKEND", "memory")
    )
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", "redis://localhost:6379/0"))
    redis_stream_maxlen: int = field(default_factory=lambda: _env_int("REDIS_STREAM_MAXLEN", 10000))
    nats_url: str = field(default_factory=lambda: _env("NATS_URL", "nats://localhost:4222"))
    nats_subject_prefix: str = field(default_factory=lambda: _env("NATS_SUBJECT_PREFIX", "director_factory"))
    request_timeout: float = field(default_factory=lambda: _env_float("MSG_TIMEOUT", 60.0))
    dead_letter_enabled: bool = field(default_factory=lambda: _env_bool("MSG_DEAD_LETTER", False))


@dataclass
class DatabaseConfig:
    """持久化配置"""
    backend: Literal["memory", "postgresql"] = field(
        default_factory=lambda: _env("DATABASE_BACKEND", "memory")  # type: ignore[arg-type]
    )
    postgres_url: str = field(default_factory=lambda: _env("DATABASE_URL", "postgresql://localhost:5432/director_factory"))
    pg_pool_min: int = field(default_factory=lambda: _env_int("PG_POOL_MIN", 2))
    pg_pool_max: int = field(default_factory=lambda: _env_int("PG_POOL_MAX", 10))
    pg_vector_dim: int = field(default_factory=lambda: _env_int("PG_VECTOR_DIM", 512))
    checkpoint_dir: str = field(default_factory=lambda: _env("CHECKPOINT_DIR", "outputs/checkpoints"))


@dataclass
class ResourceConfig:
    """资源与调度配置"""
    gpu_total_vram_gb: float = field(default_factory=lambda: _env_float("GPU_VRAM_GB", 24.0))
    gpu_vram_per_shot_gb: float = field(default_factory=lambda: _env_float("GPU_VRAM_PER_SHOT", 2.5))
    max_concurrent_shots: int = field(default_factory=lambda: _env_int("MAX_CONCURRENT_SHOTS", 8))
    max_retries_per_shot: int = field(default_factory=lambda: _env_int("MAX_RETRIES", 3))
    quality_threshold: float = field(default_factory=lambda: _env_float("QUALITY_THRESHOLD", 0.85))
    ffmpeg_path: str = field(default_factory=lambda: _env("FFMPEG_PATH", "ffmpeg"))


@dataclass
class LoggingConfig:
    """日志与监控配置"""
    level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    format: Literal["text", "json"] = field(default_factory=lambda: _env("LOG_FORMAT", "text"))
    metrics_enabled: bool = field(default_factory=lambda: _env_bool("METRICS_ENABLED", False))
    metrics_port: int = field(default_factory=lambda: _env_int("METRICS_PORT", 9090))
    trace_enabled: bool = field(default_factory=lambda: _env_bool("TRACE_ENABLED", True))
    sentry_dsn: str = field(default_factory=lambda: _env("SENTRY_DSN"))


@dataclass
class ProductionConfig:
    """顶层生产配置 — 所有子配置的聚合"""

    llm: LLMConfig = field(default_factory=LLMConfig)
    image_gen: ImageGenConfig = field(default_factory=ImageGenConfig)
    cloud_image_gen: CloudImageGenConfig = field(default_factory=CloudImageGenConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    message_bus: MessageBusConfig = field(default_factory=MessageBusConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # 运行模式
    mode: Literal["production", "staging", "development"] = "development"
    project_name: str = field(default_factory=lambda: _env("PROJECT_NAME", "director_factory"))
    output_dir: str = field(default_factory=lambda: _env("OUTPUT_DIR", "outputs"))
    mock_all: bool = field(default_factory=lambda: _env_bool("MOCK_ALL", True))

    def __post_init__(self):
        env_mode = _env("DIRECTOR_MODE", "")
        if env_mode in ("production", "staging", "development"):
            self.mode = env_mode
        if self.mode == "production":
            self.mock_all = _env_bool("MOCK_ALL", False)

    @classmethod
    def from_env(cls) -> "ProductionConfig":
        """从环境变量加载完整配置"""
        return cls(
            llm=LLMConfig(),
            image_gen=ImageGenConfig(),
            cloud_image_gen=CloudImageGenConfig(),
            tts=TTSConfig(),
            message_bus=MessageBusConfig(),
            database=DatabaseConfig(),
            resources=ResourceConfig(),
            logging=LoggingConfig(),
        )

    def to_dict(self) -> dict:
        """导出为字典 (不含敏感信息)"""
        return {
            "mode": self.mode,
            "project_name": self.project_name,
            "mock_all": self.mock_all,
            "llm": {
                "provider": self.llm.provider,
                "default_model": self.llm.default_model,
                "fallback_model": self.llm.fallback_model,
                "has_api_key": bool(self.llm.anthropic_api_key or self.llm.openai_api_key),
            },
            "image_gen": {"provider": self.image_gen.provider, "sdxl_model": self.image_gen.sdxl_model},
            "cloud_image_gen": {"provider_order": self.cloud_image_gen.provider_order},
            "tts": {"provider": self.tts.provider, "language": self.tts.tts_language},
            "message_bus": {"backend": self.message_bus.backend},
            "database": {"backend": self.database.backend},
            "resources": {
                "gpu_vram_gb": self.resources.gpu_total_vram_gb,
                "max_concurrent": self.resources.max_concurrent_shots,
                "quality_threshold": self.resources.quality_threshold,
            },
            "logging": {"level": self.logging.level, "format": self.logging.format},
            "output_dir": self.output_dir,
        }


# 全局单例
config = ProductionConfig.from_env()
