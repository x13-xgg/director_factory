"""翻译工具 — LLM-based 对话翻译 + 语言常量"""

from __future__ import annotations

import json
import os

from src.core.config import config
from src.core.logging import get_logger
from src.tools.base import BaseTool, ToolCall, ToolResult

log = get_logger("Translator")

SUPPORTED_LANGUAGES = {
    "zh": {"name": "Chinese", "edge_tts_prefix": "zh-CN", "wps": 4.0, "line_sep": "；"},
    "en": {"name": "English", "edge_tts_prefix": "en-US", "wps": 3.0, "line_sep": "."},
    "ja": {"name": "Japanese", "edge_tts_prefix": "ja-JP", "wps": 4.5, "line_sep": "。"},
    "ko": {"name": "Korean", "edge_tts_prefix": "ko-KR", "wps": 4.0, "line_sep": "."},
}

LANGUAGE_NAMES = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
}


def get_language_info(lang: str) -> dict:
    return SUPPORTED_LANGUAGES.get(lang, SUPPORTED_LANGUAGES["zh"])


def is_supported_language(lang: str) -> bool:
    return lang in SUPPORTED_LANGUAGES


class TranslationTool(BaseTool):
    """LLM-based dialog translation — single call batches all shot dialogs"""

    def __init__(self):
        super().__init__("translate")
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        api_key = os.getenv("DEEPSEEK_API_KEY", "") or config.llm.openai_api_key
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        if not api_key:
            log.warn("No API key for translation, will use mock placeholder")
            return None
        try:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=config.llm.request_timeout,
                max_retries=0,
            )
        except Exception as e:
            log.warn(f"Translation client init failed: {e}")
        return self._client

    def schema(self) -> dict:
        return {
            "name": "translate",
            "description": "Translate dialog texts to target language while preserving emotion and tone",
            "parameters": {
                "texts": {"type": "array", "items": {"type": "string"}},
                "target_language": {"type": "string", "default": "en"},
                "source_language": {"type": "string", "default": "zh"},
            },
        }

    async def execute(self, call: ToolCall) -> ToolResult:
        texts = call.params.get("texts", [])
        target_lang = call.params.get("target_language", "en")
        source_lang = call.params.get("source_language", "zh")

        if not texts:
            return ToolResult.ok(data={"translations": {}})

        if not is_supported_language(target_lang):
            return ToolResult(
                status="fail",
                data={"error": f"Unsupported language: {target_lang}"},
            )

        if target_lang == source_lang:
            translations = {t: t for t in texts}
            return ToolResult.ok(data={"translations": translations, "method": "identity"})

        client = self._get_client()
        if client is None:
            return ToolResult.ok(
                data={"translations": {t: t for t in texts}, "method": "mock"},
                mock=True,
            )

        target_name = LANGUAGE_NAMES.get(target_lang, target_lang)
        source_name = LANGUAGE_NAMES.get(source_lang, source_lang)

        system = (
            f"You are a professional translator. Translate the following dialog lines "
            f"from {source_name} to {target_name}. Preserve the emotional tone, speaking style, "
            f"and natural conversational flow. Output ONLY valid JSON."
        )

        items = "\n".join(f"{i}: {t}" for i, t in enumerate(texts))
        user = (
            f"Translate each numbered line from {source_name} to {target_name}.\n"
            f"Return a JSON object mapping the original text to its translation.\n\n"
            f"{items}"
        )

        schema = {
            "type": "object",
            "properties": {},
            "additionalProperties": {"type": "string"},
        }

        try:
            from openai import AsyncOpenAI
            response = await client.chat.completions.create(
                model="deepseek-chat",
                max_tokens=4096,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "system", "content": f"You MUST output valid JSON. The JSON object should map each original text to its translation. Example: {{\"original text 1\": \"translated text 1\", \"original text 2\": \"translated text 2\"}}"},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content or "{}"
            translations = json.loads(content)

            if not isinstance(translations, dict):
                translations = {t: t for t in texts}

            # Fill in any missing texts with originals
            for t in texts:
                if t not in translations:
                    translations[t] = t

            log.info(f"Translated {len(translations)} dialogs to {target_name}")
            return ToolResult.ok(data={
                "translations": translations,
                "method": "llm",
                "usage": {
                    "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "output_tokens": response.usage.completion_tokens if response.usage else 0,
                },
            })
        except Exception as e:
            log.warn(f"Translation failed: {e}")
            return ToolResult.ok(
                data={"translations": {t: t for t in texts}, "method": "mock"},
                mock=True,
            )
