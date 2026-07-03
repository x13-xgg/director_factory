"""音频/视频后期工具 — TTS、音效匹配、调色、字幕/VFX"""

from __future__ import annotations

import asyncio
import hashlib
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from src.tools.base import BaseTool, ToolCall, ToolResult
from src.tools.asset_db import asset_db
from src.tools.translator import get_language_info
from src.core.config import config
from src.core.logging import get_logger

log = get_logger("TTSTool")


def _get_ffmpeg() -> str:
    ffmpeg_path = config.resources.ffmpeg_path
    if ffmpeg_path == "ffmpeg":
        return "ffmpeg"
    exe = os.path.join(ffmpeg_path, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if os.path.exists(exe):
        return exe
    exe = os.path.join(ffmpeg_path, "ffmpeg")
    if os.path.exists(exe):
        return exe
    return "ffmpeg"


def _escape_ffmpeg_path(path: str) -> str:
    """转义 Windows 路径中 ffmpeg filter 的冒号和反斜杠"""
    if os.name == "nt":
        return path.replace("\\", "/").replace(":", "\\:")
    return path


class TTSTool(BaseTool):
    """TTS 语音合成 — 支持 edge_tts / bark / chattts / mock"""

    # 语言 → 情绪 → 语音角色映射
    VOICE_MAP = {
        "zh": {
            "neutral":    "zh-CN-XiaoxiaoNeural",
            "happy":      "zh-CN-XiaoxiaoNeural",
            "sad":        "zh-CN-YunyangNeural",
            "angry":      "zh-CN-YunjianNeural",
            "fear":       "zh-CN-XiaoyiNeural",
            "calm":       "zh-CN-YunxiaNeural",
            "excited":    "zh-CN-YunxiNeural",
            "gentle":     "zh-CN-YunxiaNeural",
            "serious":    "zh-CN-YunjianNeural",
            "whisper":    "zh-CN-XiaoxiaoNeural",
            "default":    "zh-CN-XiaoxiaoNeural",
        },
        "en": {
            "neutral":    "en-US-JennyNeural",
            "happy":      "en-US-JennyNeural",
            "sad":        "en-US-AriaNeural",
            "angry":      "en-US-GuyNeural",
            "fear":       "en-US-AriaNeural",
            "calm":       "en-US-JennyNeural",
            "excited":    "en-US-JennyNeural",
            "gentle":     "en-US-JennyNeural",
            "serious":    "en-US-GuyNeural",
            "whisper":    "en-US-AriaNeural",
            "default":    "en-US-JennyNeural",
        },
        "ja": {
            "neutral":    "ja-JP-NanamiNeural",
            "happy":      "ja-JP-NanamiNeural",
            "sad":        "ja-JP-NanamiNeural",
            "angry":      "ja-JP-KeitaNeural",
            "fear":       "ja-JP-NanamiNeural",
            "calm":       "ja-JP-NanamiNeural",
            "excited":    "ja-JP-NanamiNeural",
            "gentle":     "ja-JP-NanamiNeural",
            "serious":    "ja-JP-KeitaNeural",
            "whisper":    "ja-JP-NanamiNeural",
            "default":    "ja-JP-NanamiNeural",
        },
        "ko": {
            "neutral":    "ko-KR-SunHiNeural",
            "happy":      "ko-KR-SunHiNeural",
            "sad":        "ko-KR-SunHiNeural",
            "angry":      "ko-KR-InJoonNeural",
            "fear":       "ko-KR-SunHiNeural",
            "calm":       "ko-KR-SunHiNeural",
            "excited":    "ko-KR-SunHiNeural",
            "gentle":     "ko-KR-SunHiNeural",
            "serious":    "ko-KR-InJoonNeural",
            "whisper":    "ko-KR-SunHiNeural",
            "default":    "ko-KR-SunHiNeural",
        },
    }

    # 语速与音高修正 (edge_tts 使用 rate 字符串, 如 "+20%")
    EMOTION_RATE_PITCH = {
        "neutral":  ("+0%", "+0Hz"),
        "happy":    ("+10%", "+5Hz"),
        "sad":      ("-15%", "-8Hz"),
        "angry":    ("+5%", "+0Hz"),
        "fear":     ("+15%", "+10Hz"),
        "calm":     ("-10%", "-3Hz"),
        "excited":  ("+20%", "+8Hz"),
        "gentle":   ("-8%", "-2Hz"),
        "serious":  ("-5%", "-5Hz"),
        "whisper":  ("-20%", "-2Hz"),
    }

    def __init__(self):
        super().__init__("tts")
        self._chattts = None          # 懒加载 ChatTTS 模型
        self._chattts_speakers: dict = {}  # char_id → speaker embedding

    def schema(self) -> dict:
        return {
            "name": "tts",
            "description": "Generate speech audio from text with emotion control",
            "parameters": {
                "text": {"type": "string"},
                "emotion": {"type": "string", "default": "neutral"},
                "character_id": {"type": "string"},
                "voice_profile": {"type": "string", "default": "default"},
                "speed": {"type": "number", "default": 1.0},
                "pitch_variance": {"type": "number", "default": 0.0},
                "language": {"type": "string", "default": "zh"},
            },
        }

    async def execute(self, call: ToolCall) -> ToolResult:
        text = call.params.get("text", "")
        emotion = call.params.get("emotion", "neutral")
        char_id = call.params.get("character_id", "unknown")
        speed = call.params.get("speed", 1.0)
        pitch_var = call.params.get("pitch_variance", 0.0)
        voice_profile = call.params.get("voice_profile", "default")
        language = call.params.get("language", "zh")

        if not text:
            return ToolResult.ok(
                data={"audio_path": "", "duration": 0.0, "phoneme_timestamps": []},
            )

        provider = config.tts.provider

        # 非中文时 bark/chattts 自动回退到 edge_tts
        if language != "zh" and provider in ("bark", "chattts"):
            log.info(f"非中文({language})，{provider} 回退到 edge_tts")
            provider = "edge_tts"

        if provider == "edge_tts" and not config.mock_all:
            return await self._edge_tts_generate(text, emotion, char_id, speed, pitch_var, voice_profile, language)
        elif provider == "bark" and not config.mock_all:
            return await self._bark_generate(text, emotion, char_id, speed, pitch_var, voice_profile, language)
        elif provider == "chattts" and not config.mock_all:
            return await self._chattts_generate(text, emotion, char_id, speed, pitch_var, voice_profile, language)
        else:
            return self._mock_generate(text, emotion, char_id, speed, pitch_var, language)

    # ── edge_tts 后端 ─────────────────────────────────

    async def _edge_tts_generate(
        self, text: str, emotion: str, char_id: str,
        speed: float, pitch_var: float, voice_profile: str, language: str = "zh",
    ) -> ToolResult:
        try:
            import edge_tts

            voice = self._pick_voice(emotion, voice_profile, language)
            rate_str, pitch_str = self.EMOTION_RATE_PITCH.get(emotion, ("+0%", "+0Hz"))

            # 应用 speed 和 pitch 微调
            if speed != 1.0:
                base_rate = int(rate_str.strip("%Hz").replace("+", "").replace("-", "") or 0)
                adjusted = base_rate + int((speed - 1.0) * 50)
                rate_str = f"+{adjusted}%" if adjusted >= 0 else f"{adjusted}%"
            if pitch_var != 0.0:
                base_pitch = int(pitch_str.strip("%Hz").replace("+", "").replace("-", "") or 0)
                adjusted = base_pitch + int(pitch_var * 12)  # 12Hz = 1 semitone
                pitch_str = f"+{adjusted}Hz" if adjusted >= 0 else f"{adjusted}Hz"

            audio_hash = hashlib.sha256(f"{char_id}:{text[:50]}:{emotion}".encode()).hexdigest()[:12]
            out_dir = Path(config.output_dir) / "audio"
            out_dir.mkdir(parents=True, exist_ok=True)
            mp3_path = out_dir / f"{char_id}_{audio_hash}.mp3"

            # 使用 edge_tts Communicate 直接生成
            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=rate_str,
                pitch=pitch_str,
            )
            await communicate.save(str(mp3_path))

            # 按语言估算时长
            lang_info = get_language_info(language)
            is_cjk = language in ("zh", "ja")
            words = len(text) if is_cjk else len(text.split())
            rate_wps = lang_info["wps"]
            duration = max(0.5, words / rate_wps * (2.0 - speed) if speed < 1.5 else words / rate_wps / speed)

            log.info(f"TTS 生成: {char_id} [{emotion}] voice={voice}, file={mp3_path.name}")

        except Exception as e:
            log.warn(f"edge_tts 生成失败, 回退到 mock: {e}")
            return self._mock_generate(text, emotion, char_id, speed, pitch_var)

        result_data = {
            "audio_path": str(mp3_path),
            "text": text,
            "duration": round(duration, 2),
            "emotion": emotion,
            "character_id": char_id,
            "voice": voice,
            "speed": speed,
            "pitch_variance": pitch_var,
            "rate_str": rate_str,
            "pitch_str": pitch_str,
            "sample_rate": config.tts.tts_sample_rate,
            "phoneme_timestamps": [],
            "generated_at": time.time(),
            "gen_method": "edge_tts",
        }

        asset_db.put("sfx_library", f"tts:{char_id}:{audio_hash}", result_data, {"type": "tts_clip"})
        return ToolResult.ok(data=result_data)

    # ── Bark 后端 ─────────────────────────────────────

    async def _bark_generate(
        self, text: str, emotion: str, char_id: str,
        speed: float, pitch_var: float, voice_profile: str, language: str = "zh",
    ) -> ToolResult:
        audio_hash = hashlib.sha256(f"{char_id}:{text[:50]}:{emotion}".encode()).hexdigest()[:12]
        out_dir = Path(config.output_dir) / "audio"
        out_dir.mkdir(parents=True, exist_ok=True)
        wav_path = out_dir / f"{char_id}_{audio_hash}.wav"

        # 优先: 本地 suno-bark 直接生成
        local_ok = await self._bark_local(text, emotion, voice_profile, wav_path)
        if local_ok:
            return self._package_result(wav_path, text, emotion, char_id, speed, pitch_var, "bark_local", language)

        # 次选: Bark HTTP 服务器
        try:
            import base64
            import httpx

            bark_url = config.tts.bark_url.rstrip("/")
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
                resp = await client.post(f"{bark_url}/synthesize", json={
                    "text": text,
                    "voice": voice_profile or "zh_speaker_0",
                    "emotion": emotion,
                })
                resp.raise_for_status()
                data = resp.json()
            if "audio_base64" in data:
                wav_path.write_bytes(base64.b64decode(data["audio_base64"]))
            log.info(f"Bark HTTP TTS 生成: {char_id} [{emotion}], file={wav_path.name}")
            return self._package_result(wav_path, text, emotion, char_id, speed, pitch_var, "bark_http", language)
        except Exception as e:
            log.warn(f"Bark 生成失败, 回退到 mock: {e}")
            return self._mock_generate(text, emotion, char_id, speed, pitch_var)

    def _package_result(
        self, wav_path, text: str, emotion: str, char_id: str,
        speed: float, pitch_var: float, gen_method: str, language: str = "zh",
    ) -> ToolResult:
        lang_info = get_language_info(language)
        is_cjk = language in ("zh", "ja")
        words = len(text) if is_cjk else len(text.split())
        duration = max(0.5, words / lang_info["wps"] * speed)
        audio_hash = hashlib.sha256(f"{char_id}:{text[:50]}:{emotion}".encode()).hexdigest()[:12]
        result_data = {
            "audio_path": str(wav_path),
            "text": text,
            "duration": round(duration, 2),
            "emotion": emotion,
            "character_id": char_id,
            "speed": speed,
            "pitch_variance": pitch_var,
            "sample_rate": config.tts.tts_sample_rate,
            "phoneme_timestamps": [],
            "generated_at": time.time(),
            "gen_method": gen_method,
        }
        asset_db.put("sfx_library", f"tts:{char_id}:{audio_hash}", result_data, {"type": "tts_clip"})
        return ToolResult.ok(data=result_data)

    async def _bark_local(
        self, text: str, emotion: str, voice_profile: str, output_path: Path,
    ) -> bool:
        """本地 suno-bark 直接生成。成功返回 True。"""
        try:
            from bark import SAMPLE_RATE, generate_audio
            from bark.generation import preload_models
            from scipy.io.wavfile import write as write_wav

            # 首次调用时预加载模型 (~10GB, 需时较长)
            if not hasattr(self, "_bark_models_loaded"):
                preload_models(text_use_small=True, coarse_use_small=True, fine_use_gpu=False)
                self._bark_models_loaded = True

            # 选择语音预设
            is_chinese = any('一' <= c <= '鿿' for c in text)
            voice = voice_profile or ("v2/zh_speaker_3" if is_chinese else "v2/en_speaker_6")

            # 在 executor 中运行 (Bark 是同步的)
            loop = __import__('asyncio').get_running_loop()
            audio_array = await loop.run_in_executor(
                None, generate_audio, text, voice,
            )

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            await loop.run_in_executor(None, write_wav, str(output_path), SAMPLE_RATE, audio_array)
            log.info(f"Bark 本地生成: {output_path.name}")
            return True
        except Exception as e:
            log.warn(f"Bark 本地生成失败: {e}")
            return False

    # ── ChatTTS 后端 ───────────────────────────────────

    # ChatTTS 情绪 → speed prompt 映射 (speed_1=最慢, speed_5=最快)
    CHATTS_EMOTION_SPEED = {
        "neutral":  "[speed_4]",
        "happy":    "[speed_5]",
        "sad":      "[speed_2]",
        "angry":    "[speed_5]",
        "fear":     "[speed_5]",
        "calm":     "[speed_2]",
        "excited":  "[speed_5]",
        "gentle":   "[speed_3]",
        "serious":  "[speed_3]",
        "whisper":  "[speed_1]",
    }

    async def _load_chattts(self) -> bool:
        """懒加载 ChatTTS 模型。成功返回 True。"""
        if self._chattts is not None:
            return True
        try:
            from ChatTTS import Chat
            import torch

            model_path = config.tts.chattts_model_path
            custom = None
            source = "local"
            if Path(model_path).exists():
                # 查找 HF 缓存 snapshot 目录 (models--2Noise--ChatTTS/snapshots/*)
                snapshots_dir = Path(model_path) / "models--2Noise--ChatTTS" / "snapshots"
                if snapshots_dir.exists():
                    snaps = sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
                    for s in snaps:
                        if s.is_dir() and (s / "asset").exists():
                            custom = str(s)
                            source = "custom"
                            break
                if custom is None:
                    custom = model_path
                    source = "custom"
            chat = Chat()
            ok = chat.load(
                source=source,
                custom_path=custom,
                device=torch.device("cpu"),
                compile=False,
            )
            if not ok:
                log.warn("ChatTTS 模型加载失败, 将回退到 mock")
                return False
            self._chattts = chat
            ChatTTSModelHolder.set(chat)
            log.info(f"ChatTTS 模型加载成功 (source={source}, path={custom})")
            return True
        except ImportError:
            log.warn("ChatTTS 未安装, 回退 mock. pip install ChatTTS")
            return False
        except Exception as e:
            log.warn(f"ChatTTS 加载失败: {e}")
            return False

    async def _chattts_generate(
        self, text: str, emotion: str, char_id: str,
        speed: float, pitch_var: float, voice_profile: str, language: str = "zh",
    ) -> ToolResult:
        try:
            if not await self._load_chattts():
                return self._mock_generate(text, emotion, char_id, speed, pitch_var)

            audio_hash = hashlib.sha256(f"{char_id}:{text[:50]}:{emotion}".encode()).hexdigest()[:12]
            out_dir = Path(config.output_dir) / "audio"
            out_dir.mkdir(parents=True, exist_ok=True)
            wav_path = out_dir / f"{char_id}_{audio_hash}.wav"

            chat = self._chattts

            # 角色一致性: 缓存 speaker embedding
            spk_emb = self._chattts_speakers.get(char_id)
            if spk_emb is None and voice_profile == "default":
                spk_emb = chat.sample_random_speaker()
                self._chattts_speakers[char_id] = spk_emb

            # 情绪 → 速度 prompt
            speed_prompt = self.CHATTS_EMOTION_SPEED.get(emotion, "[speed_4]")
            if speed >= 1.3:
                speed_prompt = "[speed_5]"
            elif speed <= 0.7:
                speed_prompt = "[speed_2]"

            # 在 executor 中运行 (ChatTTS 是同步的)
            loop = asyncio.get_running_loop()
            params = ChatTTSInferParams(
                text=text,
                spk_emb=spk_emb,
                speed_prompt=speed_prompt,
                wav_path=str(wav_path),
            )
            duration = await loop.run_in_executor(None, self._run_chattts, params)

            lang_info = get_language_info(language)
            is_cjk = language in ("zh", "ja")
            est_duration = max(0.5, (len(text) if is_cjk else len(text.split())) / lang_info["wps"] * (2.0 - speed))
            log.info(f"ChatTTS 生成: {char_id} [{emotion}], file={wav_path.name}, dur={duration:.1f}s")

        except Exception as e:
            log.warn(f"ChatTTS 生成失败, 回退到 mock: {e}")
            return self._mock_generate(text, emotion, char_id, speed, pitch_var)

        result_data = {
            "audio_path": str(wav_path),
            "text": text,
            "duration": round(duration if duration > 0 else est_duration, 2),
            "emotion": emotion,
            "character_id": char_id,
            "speed": speed,
            "pitch_variance": pitch_var,
            "sample_rate": 24000,
            "phoneme_timestamps": [],
            "generated_at": time.time(),
            "gen_method": "chattts",
        }

        audio_hash = wav_path.stem.split("_", 1)[1] if "_" in wav_path.stem else wav_path.stem
        asset_db.put("sfx_library", f"tts:{char_id}:{audio_hash}", result_data, {"type": "tts_clip"})
        return ToolResult.ok(data=result_data)

    @staticmethod
    def _run_chattts(params) -> float:
        """在 executor 线程中运行 ChatTTS 推理 (同步)"""
        import numpy as np
        from scipy.io.wavfile import write as write_wav
        from ChatTTS import Chat

        chat = ChatTTSModelHolder.get()
        wavs = chat.infer(
            params.text,
            params_refine_text=Chat.RefineTextParams(prompt=params.speed_prompt),
            params_infer_code=Chat.InferCodeParams(
                prompt=params.speed_prompt,
                spk_emb=params.spk_emb,
                temperature=0.3,
            ),
            use_decoder=True,
            do_text_normalization=True,
            do_homophone_replacement=True,
        )
        if not wavs or len(wavs) == 0:
            return 0.0
        audio = np.concatenate(wavs) if len(wavs) > 1 else wavs[0]
        audio = (audio * 32767).astype(np.int16)
        write_wav(params.wav_path, 24000, audio)
        return len(audio) / 24000.0

    # ── Mock ──────────────────────────────────────────

    def _mock_generate(
        self, text: str, emotion: str, char_id: str,
        speed: float, pitch_var: float, language: str = "zh",
    ) -> ToolResult:
        lang_info = get_language_info(language)
        is_cjk = language in ("zh", "ja")
        words = len(text) if is_cjk else len(text.split())
        rate = lang_info["wps"]
        duration = max(0.5, words / rate * speed)

        audio_hash = hashlib.sha256(f"{char_id}:{text[:50]}:{emotion}".encode()).hexdigest()[:12]
        audio_path = f"outputs/audio/{char_id}_{audio_hash}.wav"
        Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
        Path(audio_path).touch()

        phoneme_timestamps = []
        segment_dur = duration / max(len(text), 1)
        for i, ch in enumerate(text[:20]):
            if ch.strip():
                phoneme_timestamps.append({
                    "char": ch,
                    "start_ms": round(i * segment_dur * 1000),
                    "end_ms": round((i + 1) * segment_dur * 1000),
                })

        result_data = {
            "audio_path": audio_path,
            "text": text,
            "duration": round(duration, 2),
            "emotion": emotion,
            "character_id": char_id,
            "speed": speed,
            "pitch_variance": pitch_var,
            "phoneme_timestamps": phoneme_timestamps,
            "sample_rate": 24000,
            "generated_at": time.time(),
            "gen_method": "mock",
        }

        asset_db.put("sfx_library", f"tts:{char_id}:{audio_hash}", result_data, {"type": "tts_clip"})
        return ToolResult.ok(data=result_data, mock=True)

    def _pick_voice(self, emotion: str, voice_profile: str, language: str = "zh") -> str:
        voice_map = self.VOICE_MAP.get(language, self.VOICE_MAP["zh"])
        if voice_profile and voice_profile != "default":
            if voice_profile in voice_map:
                return voice_map[voice_profile]
            return voice_profile
        return voice_map.get(emotion, voice_map["default"])


class ChatTTSInferParams:
    __slots__ = ("text", "spk_emb", "speed_prompt", "wav_path")
    def __init__(self, text, spk_emb, speed_prompt, wav_path):
        self.text = text
        self.spk_emb = spk_emb
        self.speed_prompt = speed_prompt
        self.wav_path = wav_path


class ChatTTSModelHolder:
    """线程安全的 ChatTTS 模型持有者 — 供 executor 线程使用"""
    _chat = None

    @classmethod
    def set(cls, chat):
        cls._chat = chat

    @classmethod
    def get(cls):
        if cls._chat is None:
            raise RuntimeError("ChatTTS model not loaded")
        return cls._chat


class SFXMatcherTool(BaseTool):
    """音效匹配 — 根据场景描述匹配音效"""

    SFX_CATALOG = {
        "footstep": "sfx/footstep_concrete.wav",
        "wind": "sfx/wind_ambient.wav",
        "rain": "sfx/rain_light.wav",
        "thunder": "sfx/thunder_distant.wav",
        "door_open": "sfx/door_metal_creak.wav",
        "door_close": "sfx/door_slam.wav",
        "glass_break": "sfx/glass_shatter.wav",
        "metal_clank": "sfx/metal_impact.wav",
        "engine": "sfx/engine_hum.wav",
        "footsteps_gravel": "sfx/footstep_gravel.wav",
        "water_drip": "sfx/water_drip_echo.wav",
        "fire": "sfx/fire_crackle.wav",
        "electric_buzz": "sfx/electric_hum.wav",
        "silence": "sfx/silence_room_tone.wav",
        "crowd": "sfx/crowd_murmur.wav",
        "birds": "sfx/birds_ambient.wav",
        "machinery": "sfx/machinery_low_rumble.wav",
        "heartbeat": "sfx/heartbeat_deep.wav",
        "static_noise": "sfx/static_radio.wav",
        "wind_howl": "sfx/wind_howl_gust.wav",
    }

    def __init__(self):
        super().__init__("sfx_matcher")

    def schema(self) -> dict:
        return {
            "name": "sfx_matcher",
            "description": "Match sound effects to scene descriptions",
            "parameters": {
                "action_description": {"type": "string"},
                "location": {"type": "string"},
                "mood": {"type": "string"},
                "max_results": {"type": "integer", "default": 5},
            },
        }

    async def execute(self, call: ToolCall) -> ToolResult:
        action = call.params.get("action_description", "").lower()
        location = call.params.get("location", "").lower()
        mood = call.params.get("mood", "").lower()
        max_results = call.params.get("max_results", 5)

        matched = []
        search_text = f"{action} {location} {mood}"

        for keyword, path in self.SFX_CATALOG.items():
            key_lower = keyword.replace("_", " ")
            if key_lower in search_text or keyword in search_text.replace(" ", "_"):
                matched.append({"keyword": keyword, "path": path, "confidence": 0.90})
            elif any(word in search_text for word in key_lower.split("_")):
                matched.append({"keyword": keyword, "path": path, "confidence": 0.70})

        # 如果没有匹配，给默认环境音
        if not matched:
            if "室外" in action or "outdoor" in search_text:
                matched.append({"keyword": "wind", "path": self.SFX_CATALOG["wind"], "confidence": 0.60})
            else:
                matched.append({"keyword": "silence", "path": self.SFX_CATALOG["silence"], "confidence": 0.60})

        matched.sort(key=lambda x: x["confidence"], reverse=True)
        matched = matched[:max_results]

        # Verify matched files actually exist on disk
        assets_root = Path(__file__).parent.parent.parent / "assets"
        for m in matched:
            full_path = assets_root / m["path"]
            if full_path.exists() and full_path.stat().st_size > 0:
                m["available"] = True
                m["absolute_path"] = str(full_path.resolve())
            else:
                m["available"] = False

        return ToolResult.ok(
            data={
                "sfx_matches": matched,
                "total_matches": len(matched),
                "scene_context": action[:100],
            },
        )


class BGMMatcherTool(BaseTool):
    """BGM 匹配 — 根据情绪/场景选择背景音乐"""

    BGM_CATALOG = {
        "tension": {"path": "assets/bgm/tension_bed.wav", "bpm": 90, "key": "Dm", "style": "dark ambient"},
        "hope": {"path": "assets/bgm/hope_theme.wav", "bpm": 72, "key": "C", "style": "orchestral"},
        "sadness": {"path": "assets/bgm/sad_strings.wav", "bpm": 60, "key": "Am", "style": "solo strings"},
        "joy": {"path": "assets/bgm/joy_theme.wav", "bpm": 120, "key": "G", "style": "upbeat pop"},
        "fear": {"path": "assets/bgm/fear_drone.wav", "bpm": 50, "key": "C#m", "style": "drone ambient"},
        "action": {"path": "assets/bgm/action_beat.wav", "bpm": 140, "key": "Em", "style": "electronic"},
        "loneliness": {"path": "assets/bgm/lonely_piano.wav", "bpm": 66, "key": "Fm", "style": "solo piano"},
        "serene": {"path": "assets/bgm/serene_pad.wav", "bpm": 70, "key": "Eb", "style": "ambient pad"},
        "wistful": {"path": "assets/bgm/wistful_guitar.wav", "bpm": 78, "key": "D", "style": "acoustic"},
        "neutral": {"path": "assets/bgm/neutral_bed.wav", "bpm": 85, "key": "C", "style": "light ambient"},
    }

    def __init__(self):
        super().__init__("bgm_matcher")

    def schema(self) -> dict:
        return {
            "name": "bgm_matcher",
            "description": "Select background music based on emotion and scene context",
            "parameters": {
                "emotion": {"type": "string", "default": "neutral"},
                "scene_mood": {"type": "string"},
                "tempo_hint": {"type": "string"},
                "intensity": {"type": "number", "default": 0.5},
            },
        }

    async def execute(self, call: ToolCall) -> ToolResult:
        emotion = call.params.get("emotion", "neutral").lower()
        scene_mood = call.params.get("scene_mood", "").lower()
        intensity = call.params.get("intensity", 0.5)

        bgm = self.BGM_CATALOG.get(emotion, self.BGM_CATALOG["neutral"])

        # 根据 intensity 调整
        adjusted_bpm = int(bgm["bpm"] * (0.8 + intensity * 0.4))

        assets_root = Path(__file__).parent.parent.parent
        full_path = assets_root / bgm["path"]
        available = full_path.exists() and full_path.stat().st_size > 0

        return ToolResult.ok(
            data={
                "bgm_path": bgm["path"],
                "absolute_path": str(full_path.resolve()) if available else None,
                "available": available,
                "bpm": adjusted_bpm,
                "key": bgm["key"],
                "style": bgm["style"],
                "emotion": emotion,
                "intensity": intensity,
                "loop": emotion in ("tension", "fear", "serene", "neutral"),
            },
        )


class ColorGradeTool(BaseTool):
    """调色工具 — 场景级色彩匹配 + LUT 生成"""

    LUT_PRESETS = {
        "desolate": "assets/luts/desolate_cold.cube",
        "warm": "assets/luts/warm_golden.cube",
        "cold": "assets/luts/cold_blue.cube",
        "cinematic": "assets/luts/cinematic_teal_orange.cube",
        "dark_fantasy": "assets/luts/dark_desaturated.cube",
        "vibrant": "assets/luts/vibrant_pop.cube",
        "noir": "assets/luts/film_noir_bw.cube",
        "sunset": "assets/luts/sunset_warm.cube",
        "industrial": "assets/luts/industrial_steel.cube",
        "neutral": "assets/luts/neutral_grade.cube",
    }

    def __init__(self):
        super().__init__("color_grade")

    def schema(self) -> dict:
        return {
            "name": "color_grade",
            "description": "Apply color grading and generate LUT for a shot or scene",
            "parameters": {
                "image_path": {"type": "string"},
                "palette_dominant": {"type": "string"},
                "palette_accent": {"type": "string"},
                "mood_descriptor": {"type": "string"},
                "color_temp_k": {"type": "integer", "default": 5600},
                "scene_id": {"type": "string"},
            },
        }

    async def execute(self, call: ToolCall) -> ToolResult:
        mood = call.params.get("mood_descriptor", "").lower()
        palette = call.params.get("palette_dominant", "").lower()
        color_temp = call.params.get("color_temp_k", 5600)
        scene_id = call.params.get("scene_id", "")
        image_path = call.params.get("image_path", "")

        # 匹配 LUT 预设
        lut_path = self.LUT_PRESETS["neutral"]
        for keyword, path in self.LUT_PRESETS.items():
            if keyword in mood or keyword in palette:
                lut_path = path
                break

        # 色温校正
        temp_correction = (color_temp - 5600) / 100

        # 色彩参数
        exposure = round(0.0 + temp_correction * 0.02, 2)
        contrast = round(1.05 + (0.5 if "dark" in mood or "noir" in mood else 0), 2)
        saturation = round(1.0 + (0.1 if "vibrant" in palette or "vibrant" in mood else -0.15 if "desolate" in mood else 0), 2)

        grade_params = {
            "exposure": exposure,
            "contrast": contrast,
            "saturation": saturation,
            "temperature": round(temp_correction, 2),
            "tint": 0.0,
            "shadows_hue": 0.0,
            "highlights_hue": 0.0,
            "lut_path": lut_path,
        }

        # 真实 LUT 应用 via ffmpeg
        if image_path and Path(image_path).exists() and not config.mock_all:
            try:
                output_dir = Path(config.output_dir) / "graded"
                output_dir.mkdir(parents=True, exist_ok=True)
                out_name = f"{Path(image_path).stem}_graded{Path(image_path).suffix}"
                out_path = output_dir / out_name

                self._apply_grade_ffmpeg(
                    str(Path(image_path).resolve()),
                    str(out_path.resolve()),
                    lut_path,
                    exposure, contrast, saturation,
                )
                image_path = str(out_path)
            except Exception as e:
                log.warn(f"ffmpeg 调色失败, 保留原始图像: {e}")

        return ToolResult.ok(
            data={
                "scene_id": scene_id,
                "image_path": image_path,
                "grade_params": grade_params,
                "lut_applied": lut_path,
                "color_temp_source": color_temp,
            },
        )

    def _apply_grade_ffmpeg(
        self, src: str, dst: str, lut: str,
        exposure: float, contrast: float, saturation: float,
    ):
        lut_abs = str(Path(lut).resolve()) if Path(lut).exists() else ""
        eq_parts = []
        if exposure != 0:
            eq_parts.append(f"brightness={exposure}")
        if contrast != 1.0:
            eq_parts.append(f"contrast={contrast}")
        if saturation != 1.0:
            eq_parts.append(f"saturation={saturation}")

        vf_filters = []
        if eq_parts:
            vf_filters.append(f"eq={':'.join(eq_parts)}")
        if lut_abs:
            vf_filters.append(f"lut3d=file='{_escape_ffmpeg_path(lut_abs)}'")

        if not vf_filters:
            return

        filter_chain = ",".join(vf_filters)
        subprocess.run(
            [_get_ffmpeg(), "-y", "-i", src, "-vf", filter_chain, "-update", "1", dst],
            capture_output=True, timeout=120,
            check=False,
        )


class VFXSubtitleTool(BaseTool):
    """字幕生成 + VFX 叠加"""

    MOOD_VFX_MAP = {
        "tension": "film_grain",
        "fear": "vignette_dark",
        "joy": "light_leak",
        "loneliness": "dust_particles",
        "sadness": "subtle_blur",
    }

    def __init__(self):
        super().__init__("vfx_subtitle")

    def schema(self) -> dict:
        return {
            "name": "vfx_subtitle",
            "description": "Generate subtitles from dialog and apply VFX overlays",
            "parameters": {
                "dialog": {"type": "string"},
                "shot_id": {"type": "string"},
                "duration": {"type": "number", "default": 3.0},
                "vfx_type": {"type": "string", "default": "none"},
                "scene_mood": {"type": "string"},
                "language": {"type": "string", "default": "zh"},
                "image_path": {"type": "string", "default": ""},
            },
        }

    async def execute(self, call: ToolCall) -> ToolResult:
        dialog = call.params.get("dialog", "")
        shot_id = call.params.get("shot_id", "")
        duration = call.params.get("duration", 3.0)
        vfx_type = call.params.get("vfx_type", "none")
        scene_mood = call.params.get("scene_mood", "")
        image_path = call.params.get("image_path", "")
        language = call.params.get("language", "zh")

        lang_info = get_language_info(language)
        line_sep = lang_info["line_sep"]

        # 生成 SRT 格式字幕
        srt_content = ""
        if dialog:
            lines = dialog.split(line_sep) if line_sep in dialog else dialog.split("；" if "；" in dialog else ";")
            if len(lines) == 1:
                lines = [dialog]
            start_time = 0.0
            line_duration = duration / max(len(lines), 1)
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                start_sec = start_time + i * line_duration
                end_sec = start_sec + line_duration
                srt_content += f"{i + 1}\n"
                srt_content += f"{self._fmt_time(start_sec)} --> {self._fmt_time(end_sec)}\n"
                srt_content += f"{line}\n\n"

        # 保存 SRT 文件
        srt_path = ""
        if srt_content:
            srt_dir = Path(config.output_dir)
            srt_dir.mkdir(parents=True, exist_ok=True)
            srt_path = str(srt_dir / f"{shot_id}_subtitle.srt")
            Path(srt_path).write_text(srt_content, encoding="utf-8")

        # 根据 mood 自动推荐 VFX
        auto_vfx = self.MOOD_VFX_MAP.get(scene_mood.lower(), "")
        actual_vfx = vfx_type if vfx_type != "none" else auto_vfx

        # VFX 参数 + 真实叠加
        vfx_params = {}
        output_image = ""
        if actual_vfx and image_path and Path(image_path).exists() and not config.mock_all:
            vfx_overlay = f"assets/vfx/{actual_vfx}_overlay.png"
            if Path(vfx_overlay).exists():
                try:
                    output_dir = Path(config.output_dir) / "vfx"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    out_name = f"{Path(image_path).stem}_vfx{Path(image_path).suffix}"
                    output_image = str(output_dir / out_name)
                    self._apply_vfx_ffmpeg(str(Path(image_path).resolve()), output_image, vfx_overlay, srt_path)
                except Exception as e:
                    log.warn(f"VFX/字幕叠加失败: {e}")

        vfx_params = {
            "type": actual_vfx,
            "intensity": 0.5,
            "duration": duration,
            "overlay_path": f"assets/vfx/{actual_vfx}_overlay.png" if actual_vfx else "",
        }

        return ToolResult.ok(
            data={
                "shot_id": shot_id,
                "srt_content": srt_content,
                "srt_path": srt_path,
                "subtitle_count": srt_content.count("\n\n") if srt_content else 0,
                "dialog_text": dialog,
                "duration": duration,
                "vfx_params": vfx_params,
                "auto_vfx_suggestion": auto_vfx,
                "output_image": output_image or image_path,
                "language": language,
            },
        )

    def _apply_vfx_ffmpeg(self, src: str, dst: str, overlay: str, srt: str):
        vf_parts = []
        if Path(overlay).exists():
            ov_escaped = _escape_ffmpeg_path(str(Path(overlay).resolve()))
            vf_parts.append(f"movie='{ov_escaped}'[ov]")
            vf_parts.append(f"[0:v][ov]overlay=0:0:format=auto")

        if srt and Path(srt).exists():
            srt_escaped = _escape_ffmpeg_path(str(Path(srt).resolve()))
            vf_parts.append(f"subtitles='{srt_escaped}'")

        if vf_parts:
            filter_chain = ";".join(vf_parts)
            subprocess.run(
                [_get_ffmpeg(), "-y", "-i", src, "-vf", filter_chain, dst],
                capture_output=True, timeout=120,
                check=False,
            )

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds * 1000) % 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
