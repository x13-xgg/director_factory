"""text_gen 工具 — 多 Provider LLM 文本生成引擎 (生产级)"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from src.tools.base import BaseTool, ToolCall, ToolResult
from src.core.config import config
from src.core.logging import get_logger

log = get_logger("TextGenTool")


class TextGenTool(BaseTool):
    """
    生产级文本生成工具 — 多 Provider 支持、自动回退、指数退避重试

    Provider 优先级:
      - 指定 provider: 使用该 provider 的 default_model → fallback_model → 其他 provider → mock
      - auto: 按可用 API key 自动选择 (anthropic > openai > deepseek > mock)
    """

    def __init__(self):
        super().__init__("text_gen")
        self._anthropic_client = None
        self._openai_client = None
        self._deepseek_client = None
        self._init_errors: dict[str, str] = {}

    # ── Lazy client initialization ──────────────────────

    def _get_anthropic_client(self):
        if self._anthropic_client is not None:
            return self._anthropic_client
        if not config.llm.anthropic_api_key:
            self._init_errors["anthropic"] = "No ANTHROPIC_API_KEY"
            self._anthropic_client = False
            return None
        try:
            from anthropic import AsyncAnthropic
            self._anthropic_client = AsyncAnthropic(
                api_key=config.llm.anthropic_api_key,
                base_url=config.llm.anthropic_base_url,
                timeout=config.llm.request_timeout,
                max_retries=0,  # we handle retries ourselves
            )
            log.info("Anthropic client initialized")
        except Exception as e:
            self._init_errors["anthropic"] = str(e)
            self._anthropic_client = False
        return self._anthropic_client if self._anthropic_client is not False else None

    def _get_openai_client(self):
        if self._openai_client is not None:
            return self._openai_client
        if not config.llm.openai_api_key:
            self._init_errors["openai"] = "No OPENAI_API_KEY"
            self._openai_client = False
            return None
        try:
            from openai import AsyncOpenAI
            self._openai_client = AsyncOpenAI(
                api_key=config.llm.openai_api_key,
                base_url=config.llm.openai_base_url,
                timeout=config.llm.request_timeout,
                max_retries=0,
            )
            log.info("OpenAI client initialized")
        except Exception as e:
            self._init_errors["openai"] = str(e)
            self._openai_client = False
        return self._openai_client if self._openai_client is not False else None

    def _get_deepseek_client(self):
        if self._deepseek_client is not None:
            return self._deepseek_client
        # 如果 OpenAI client 已指向 DeepSeek，直接复用
        if "deepseek" in config.llm.openai_base_url.lower() and config.llm.openai_api_key:
            self._deepseek_client = self._get_openai_client()
            if self._deepseek_client:
                return self._deepseek_client
        import os
        api_key = os.getenv("DEEPSEEK_API_KEY", "") or config.llm.openai_api_key
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        if not api_key:
            self._init_errors["deepseek"] = "No DEEPSEEK_API_KEY"
            self._deepseek_client = False
            return None
        try:
            from openai import AsyncOpenAI
            self._deepseek_client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=config.llm.request_timeout,
                max_retries=0,
            )
            log.info("DeepSeek client initialized")
        except Exception as e:
            self._init_errors["deepseek"] = str(e)
            self._deepseek_client = False
        return self._deepseek_client if self._deepseek_client is not False else None

    # ── Tool schema ─────────────────────────────────────

    def schema(self) -> dict:
        return {
            "name": "text_gen",
            "description": "Generate structured text with a specified output schema",
            "parameters": {
                "model": {"type": "string", "default": config.llm.default_model},
                "system_prompt": {"type": "string"},
                "user_prompt": {"type": "string"},
                "output_schema": {"type": "object"},
                "temperature": {"type": "number", "default": config.llm.temperature},
                "max_tokens": {"type": "integer", "default": config.llm.max_tokens},
            },
        }

    # ── Main execute ────────────────────────────────────

    async def execute(self, call: ToolCall) -> ToolResult:
        p = call.params
        system = p.get("system_prompt", "")
        user = p.get("user_prompt", "")
        schema = p.get("output_schema")
        temperature = p.get("temperature", config.llm.temperature)
        max_tokens = p.get("max_tokens", config.llm.max_tokens)
        model_override = p.get("model", "")
        model = model_override if model_override else config.llm.default_model

        # 全局 mock 模式
        if config.mock_all:
            log.info("mock_all=True, 使用 mock 生成")
            return self._mock_generate(user, schema)

        # 构建 fallback 链
        chain = self._build_fallback_chain(model)

        last_error = None
        for step, (provider, try_model) in enumerate(chain):
            log.info(f"尝试 provider={provider} model={try_model} (step {step + 1}/{len(chain)})")
            try:
                result = await self._call_with_retry(
                    provider, try_model, system, user, schema, temperature, max_tokens
                )
                if result is not None:
                    return result
            except Exception as e:
                last_error = e
                log.warn(f"Provider {provider}/{try_model} 失败: {e}")

        # 所有 provider 都失败 → mock
        log.warn(f"所有 provider 失败，回退到 mock。最后错误: {last_error}")
        return self._mock_generate(user, schema)

    # ── Fallback chain ──────────────────────────────────

    def _build_fallback_chain(self, requested_model: str) -> list[tuple[str, str]]:
        """构建 (provider, model) 回退链，按优先级排列"""
        chain: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        provider_pref = config.llm.provider

        def add(prov: str, mod: str):
            key = (prov, mod)
            if key not in seen:
                seen.add(key)
                chain.append(key)

        # 检测每个 provider 是否可用
        available: dict[str, bool] = {}
        if provider_pref in ("auto", "anthropic"):
            c = self._get_anthropic_client()
            available["anthropic"] = c is not None and c is not False
        if provider_pref in ("auto", "openai"):
            c = self._get_openai_client()
            available["openai"] = c is not None and c is not False
        if provider_pref in ("auto", "deepseek"):
            c = self._get_deepseek_client()
            available["deepseek"] = c is not None and c is not False

        # 首选 provider
        primary = self._resolve_primary_provider(provider_pref, available)

        # 该 provider 支持的模型列表
        def _valid_models(prov: str) -> list[str]:
            """返回某 provider 实际可用的模型"""
            if prov == "anthropic":
                return [requested_model] if requested_model.startswith("claude-") else ["claude-sonnet-4-6"]
            is_deepseek_endpoint = "deepseek" in config.llm.openai_base_url.lower()
            if is_deepseek_endpoint or prov == "deepseek":
                return ["deepseek-chat", "deepseek-v4-pro", "deepseek-v4-flash"]
            return [requested_model, "gpt-4o", "gpt-4o-mini"]

        if primary:
            for m in _valid_models(primary):
                if m not in [mod for _, mod in chain]:
                    add(primary, m)

        # 备选 provider
        for alt_provider in self._alt_providers(primary, available):
            for m in _valid_models(alt_provider):
                if m not in [mod for _, mod in chain]:
                    add(alt_provider, m)

        return chain

    def _resolve_primary_provider(self, pref: str, available: dict[str, bool]) -> str | None:
        if pref == "anthropic" and available.get("anthropic"):
            return "anthropic"
        if pref == "openai" and available.get("openai"):
            return "openai"
        if pref == "deepseek" and available.get("deepseek"):
            return "deepseek"
        if pref == "auto":
            for p in ["anthropic", "openai", "deepseek"]:
                if available.get(p):
                    return p
        return None

    def _alt_providers(self, primary: str | None, available: dict[str, bool]) -> list[str]:
        order = ["anthropic", "openai", "deepseek"]
        return [p for p in order if p != primary and available.get(p)]

    # ── Retry wrapper ───────────────────────────────────

    async def _call_with_retry(
        self,
        provider: str,
        model: str,
        system: str,
        user: str,
        schema: dict | None,
        temperature: float,
        max_tokens: int,
    ) -> ToolResult | None:
        """单个 provider 内带指数退避的重试"""
        max_retries = config.llm.max_retries
        backoff = config.llm.retry_backoff

        last_exc = None
        for attempt in range(max_retries):
            try:
                if provider == "anthropic":
                    return await self._call_anthropic(model, system, user, schema, temperature, max_tokens)
                elif provider in ("openai", "deepseek"):
                    return await self._call_openai_compatible(provider, model, system, user, schema, temperature, max_tokens)
            except Exception as e:
                last_exc = e
                if attempt < max_retries - 1:
                    wait = backoff * (2 ** attempt)
                    log.warn(f"重试 {attempt + 1}/{max_retries}, 等待 {wait:.1f}s: {e}")
                    await asyncio.sleep(wait)

        raise last_exc or RuntimeError(f"Unknown provider: {provider}")

    # ── Anthropic API call ──────────────────────────────

    async def _call_anthropic(
        self, model: str, system: str, user: str, schema: dict | None,
        temperature: float, max_tokens: int,
    ) -> ToolResult:
        client = self._get_anthropic_client()
        if client is None or client is False:
            raise RuntimeError("Anthropic client not available")

        messages = []
        if system:
            messages.append({"role": "user", "content": user})
            system_prompt = system
        else:
            messages.append({"role": "user", "content": user})
            system_prompt = None

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        if schema:
            kwargs["tools"] = [{
                "name": "output",
                "description": "Output following the schema",
                "input_schema": schema,
            }]
            kwargs["tool_choice"] = {"type": "tool", "name": "output"}

        response = await client.messages.create(**kwargs)

        usage = {"input_tokens": 0, "output_tokens": 0}
        if hasattr(response, "usage"):
            usage["input_tokens"] = response.usage.input_tokens
            usage["output_tokens"] = response.usage.output_tokens

        content = ""
        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                content = json.dumps(block.input, ensure_ascii=False)

        parsed = self._parse_output(content, schema)
        return ToolResult.ok(data=parsed or content, model=model, usage=usage, provider="anthropic")

    # ── OpenAI-compatible API call ──────────────────────

    async def _call_openai_compatible(
        self, provider: str, model: str, system: str, user: str,
        schema: dict | None, temperature: float, max_tokens: int,
    ) -> ToolResult:
        if provider == "deepseek":
            client = self._get_deepseek_client()
        else:
            client = self._get_openai_client()

        if client is None or client is False:
            raise RuntimeError(f"{provider} client not available")

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        # DeepSeek 模型的 tool_choice 支持不稳定 (包括推理模型和 chat 模型)，
        # 统一使用 json_object 模式 + prompt 内嵌 schema 绕过
        is_deepseek = provider == "deepseek" or model.startswith("deepseek-")

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }

        if schema:
            if is_deepseek:
                kwargs["response_format"] = {"type": "json_object"}
                schema_hint = json.dumps(schema, ensure_ascii=False)
                messages.append({
                    "role": "system",
                    "content": f"你必须严格按照以下 JSON Schema 输出，只输出 JSON，不要有其他文字:\n{schema_hint}",
                })
            else:
                kwargs["tools"] = [{
                    "type": "function",
                    "function": {
                        "name": "output",
                        "description": "Output following the schema",
                        "parameters": schema,
                    },
                }]
                kwargs["tool_choice"] = {"type": "function", "function": {"name": "output"}}

        response = await client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        usage = {
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
        }

        content = ""
        msg = choice.message
        if msg.content:
            content = msg.content
        if msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.function.name == "output":
                    content = tc.function.arguments
                    break

        parsed = self._parse_output(content, schema)
        return ToolResult.ok(data=parsed or content, model=model, usage=usage, provider=provider)

    # ── Output parsing ──────────────────────────────────

    def _parse_output(self, content: str, schema: dict | None) -> dict | str | None:
        """尝试将输出解析为 JSON，失败时返回原始文本"""
        if not content:
            return None
        if schema:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                if "```json" in content:
                    part = content.split("```json")[1].split("```")[0]
                    try:
                        return json.loads(part)
                    except json.JSONDecodeError:
                        pass
                return None
        return content

    # ── Mock generation ─────────────────────────────────

    def _mock_generate(self, prompt: str, schema: dict | None) -> ToolResult:
        if not schema:
            return ToolResult.ok(data={"text": prompt, "_mock": True}, mock=True)

        mock_data = self._generate_mock_from_schema(schema)
        return ToolResult.ok(data=mock_data, mock=True)

    def _generate_mock_from_schema(self, schema: dict) -> dict:
        props = schema.get("properties", {})
        required = schema.get("required", [])
        mock_data = {}

        for key, prop in props.items():
            ptype = prop.get("type", "string")
            is_required = key in required

            if ptype == "string":
                if "enum" in prop:
                    mock_data[key] = prop["enum"][0]
                elif "description" in prop:
                    mock_data[key] = prop["description"]
                else:
                    mock_data[key] = f"[mock_{key}]"

            elif ptype == "array":
                items_schema = prop.get("items", {})
                count = 3 if is_required else 1
                if items_schema.get("type") == "object":
                    mock_data[key] = [self._generate_mock_from_schema(items_schema) for _ in range(count)]
                elif items_schema.get("type") == "string":
                    if "enum" in items_schema:
                        mock_data[key] = items_schema["enum"][:count]
                    else:
                        mock_data[key] = [f"[mock_{key}_{i}]" for i in range(count)]
                elif items_schema.get("type") == "number":
                    mock_data[key] = [0.5 * (i + 1) for i in range(count)]
                else:
                    mock_data[key] = []

            elif ptype == "object":
                mock_data[key] = self._generate_mock_from_schema(prop)

            elif ptype == "number":
                mock_data[key] = 0.5 if is_required else 0.0

            elif ptype == "integer":
                mock_data[key] = 1 if is_required else 0

            elif ptype == "boolean":
                mock_data[key] = is_required

            else:
                mock_data[key] = None

        return mock_data
