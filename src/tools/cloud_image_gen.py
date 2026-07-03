"""cloud_image_gen — 多 Provider 回退链: ComfyUI → RunPod → Replicate → Modal → Mock"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from src.tools.base import BaseTool, ToolCall, ToolResult
from src.core.config import config
from src.core.logging import get_logger

log = get_logger("CloudImageGen")

PROVIDER_TIMEOUT = 300.0
MAX_RETRIES_PER_PROVIDER = 2
RETRY_BACKOFF = 2.0


class CloudImageGenTool(BaseTool):
    """
    多 Provider 图像生成工具 — 按优先级回退:

      comfyui  → 本地 ComfyUI SDXL (最快)
      runpod   →  RunPod Serverless (按秒计费)
      replicate → Replicate.com (按次计费)
      modal    →  Modal.com 自定义端点
      mock     →  本地占位图 (兜底)

    LoRA 支持: 通过 lora_path + trigger_word 参数注入 LoRA 节点
    """

    def __init__(self):
        super().__init__("cloud_image_gen")
        self._http: httpx.AsyncClient | None = None

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(PROVIDER_TIMEOUT))
        return self._http

    def schema(self) -> dict:
        return {
            "name": "cloud_image_gen",
            "description": "Generate an image via multi-provider fallback chain with LoRA support",
            "parameters": {
                "prompt": {"type": "string"},
                "negative_prompt": {"type": "string", "default": ""},
                "width": {"type": "integer", "default": config.image_gen.default_width},
                "height": {"type": "integer", "default": config.image_gen.default_height},
                "steps": {"type": "integer", "default": config.image_gen.default_steps},
                "cfg": {"type": "number", "default": config.image_gen.default_cfg},
                "seed": {"type": "integer", "default": -1},
                "batch_size": {"type": "integer", "default": config.image_gen.batch_size},
                "output_dir": {"type": "string", "default": "outputs/frames"},
                "filename": {"type": "string", "default": ""},
                "lora_path": {"type": "string", "default": ""},
                "trigger_word": {"type": "string", "default": ""},
            },
        }

    async def execute(self, call: ToolCall) -> ToolResult:
        p = call.params
        prompt = p.get("prompt", "")
        negative = p.get("negative_prompt", "")
        width = p.get("width", config.image_gen.default_width)
        height = p.get("height", config.image_gen.default_height)
        steps = p.get("steps", config.image_gen.default_steps)
        cfg = p.get("cfg", config.image_gen.default_cfg)
        seed = p.get("seed", -1)
        batch = p.get("batch_size", 1)
        output_dir = Path(p.get("output_dir", "outputs/frames"))
        filename = p.get("filename", "")
        lora_path = p.get("lora_path", "")
        trigger_word = p.get("trigger_word", "")

        if not prompt:
            return ToolResult.fail("prompt is required")

        output_dir.mkdir(parents=True, exist_ok=True)
        if seed == -1:
            seed = hash(uuid.uuid4()) & 0x7FFFFFFF

        # 确定可用 provider 链
        chain = self._build_provider_chain()

        last_error = None
        for provider in chain:
            result = await self._try_provider(
                provider, prompt, negative, width, height, steps, cfg,
                seed, batch, output_dir, filename, lora_path, trigger_word,
            )
            if result is not None:
                return result
            last_error = f"{provider} failed"

        # 最终兜底 mock
        return self._mock_generate(
            prompt, negative, width, height, output_dir, filename, seed,
        )

    # ── Provider chain ────────────────────────────────

    def _build_provider_chain(self) -> list[str]:
        """根据配置和可用性构建 Provider 回退链"""
        configured = config.cloud_image_gen.provider_order
        available = []

        for p in configured:
            if p == "comfyui":
                available.append("comfyui")
            elif p == "runpod" and config.cloud_image_gen.runpod_api_key:
                available.append("runpod")
            elif p == "replicate" and config.cloud_image_gen.replicate_api_key:
                available.append("replicate")
            elif p == "modal" and config.cloud_image_gen.modal_api_key:
                available.append("modal")
            elif p == "mock":
                available.append("mock")

        return available or ["mock"]

    async def _try_provider(
        self, provider: str, prompt: str, negative: str,
        width: int, height: int, steps: int, cfg: float,
        seed: int, batch: int, output_dir: Path, filename: str,
        lora_path: str, trigger_word: str,
    ) -> ToolResult | None:
        """尝试一个 provider，最多重试 MAX_RETRIES_PER_PROVIDER 次"""
        for attempt in range(MAX_RETRIES_PER_PROVIDER):
            try:
                if provider == "comfyui":
                    result = await self._call_comfyui(
                        prompt, negative, width, height, steps, cfg,
                        seed, batch, output_dir, filename, lora_path, trigger_word,
                    )
                elif provider == "runpod":
                    result = await self._call_runpod(
                        prompt, negative, width, height, steps, cfg, seed, batch, output_dir, filename,
                    )
                elif provider == "replicate":
                    result = await self._call_replicate(
                        prompt, negative, width, height, steps, cfg, seed, batch, output_dir, filename,
                    )
                elif provider == "modal":
                    result = await self._call_modal(
                        prompt, negative, width, height, steps, cfg, seed, batch, output_dir, filename,
                    )
                elif provider == "mock":
                    result = self._mock_generate(
                        prompt, negative, width, height, output_dir, filename, seed,
                    )
                else:
                    return None

                if result is not None:
                    return result

            except httpx.ConnectError:
                log.info(f"Provider {provider} 不可达 (attempt {attempt + 1})")
            except httpx.TimeoutException:
                log.info(f"Provider {provider} 超时 (attempt {attempt + 1})")
            except Exception as e:
                log.warn(f"Provider {provider} 错误 (attempt {attempt + 1}): {e}")

            if attempt < MAX_RETRIES_PER_PROVIDER - 1:
                await asyncio.sleep(RETRY_BACKOFF ** attempt)

        return None

    # ── ComfyUI ───────────────────────────────────────

    async def _call_comfyui(
        self, prompt: str, negative: str,
        width: int, height: int, steps: int, cfg: float,
        seed: int, batch: int, output_dir: Path, filename: str,
        lora_path: str = "", trigger_word: str = "",
    ) -> ToolResult | None:
        comfyui_url = config.cloud_image_gen.comfyui_url.rstrip("/")

        workflow = self._build_comfyui_workflow(
            prompt, negative, width, height, steps, cfg, seed, batch,
            lora_path, trigger_word,
        )

        async with httpx.AsyncClient(base_url=comfyui_url, timeout=httpx.Timeout(300.0)) as client:
            # 提交
            resp = await client.post("/prompt", json={"prompt": workflow})
            resp.raise_for_status()
            data = resp.json()
            prompt_id = data.get("prompt_id")
            if not prompt_id:
                raise RuntimeError(f"ComfyUI 返回异常: {data}")

            log.info(f"ComfyUI 任务已提交: {prompt_id}")

            # 等待完成
            deadline = time.time() + 300.0
            images = []
            while time.time() < deadline:
                try:
                    hist_resp = await client.get(f"/history/{prompt_id}")
                    hist_resp.raise_for_status()
                    history = hist_resp.json()
                    if prompt_id in history:
                        images = self._extract_comfyui_images(comfyui_url, history[prompt_id], client)
                        break
                except Exception:
                    pass
                await asyncio.sleep(1.0)

            if not images:
                return None

            saved = self._save_images(images, output_dir, filename, batch)

            return ToolResult.ok(
                data={
                    "images": saved,
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "cfg": cfg,
                    "gen_method": "comfyui_sdxl",
                },
                metadata={"prompt": prompt[:200], "seed": seed, "provider": "comfyui"},
            )

    def _extract_comfyui_images(
        self, base_url: str, history_entry: dict, client: httpx.AsyncClient,
    ) -> list[bytes]:
        images = []
        outputs = history_entry.get("outputs", {})
        for node_output in outputs.values():
            for item in node_output.get("images", []):
                filename = item.get("filename", "")
                subfolder = item.get("subfolder", "")
                img_type = item.get("type", "output")
                import urllib.parse
                params = {"filename": filename, "type": img_type}
                if subfolder:
                    params["subfolder"] = subfolder
                try:
                    # 同步下载 — 使用 httpx 同步客户端以避免嵌套事件循环
                    import urllib.request
                    qs = urllib.parse.urlencode(params)
                    full_url = f"{base_url}/view?{qs}"
                    with urllib.request.urlopen(full_url, timeout=30) as r:
                        images.append(r.read())
                except Exception:
                    pass
        return images

    # ── RunPod ────────────────────────────────────────

    async def _call_runpod(
        self, prompt: str, negative: str,
        width: int, height: int, steps: int, cfg: float,
        seed: int, batch: int, output_dir: Path, filename: str,
    ) -> ToolResult | None:
        api_key = config.cloud_image_gen.runpod_api_key
        endpoint_id = config.cloud_image_gen.runpod_endpoint_id
        if not api_key or not endpoint_id:
            return None

        url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "input": {
                "prompt": prompt,
                "negative_prompt": negative or "low quality, blurry",
                "width": width,
                "height": height,
                "num_inference_steps": steps,
                "guidance_scale": cfg,
                "seed": seed,
                "num_images": batch,
            },
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(PROVIDER_TIMEOUT)) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") != "COMPLETED":
                log.warn(f"RunPod 返回非完成状态: {data.get('status')}")
                return None

            output = data.get("output", {})
            image_urls = output.get("images", [])
            if isinstance(output, list):
                image_urls = output
            if isinstance(output, str):
                image_urls = [output]

            if not image_urls:
                return None

            images = await self._download_images(image_urls, client)
            saved = self._save_images(images, output_dir, filename, batch)

            return ToolResult.ok(
                data={
                    "images": saved,
                    "seed": seed,
                    "width": width,
                    "height": height,
                    "gen_method": "runpod",
                },
                metadata={"prompt": prompt[:200], "seed": seed, "provider": "runpod"},
            )

    # ── Replicate ─────────────────────────────────────

    async def _call_replicate(
        self, prompt: str, negative: str,
        width: int, height: int, steps: int, cfg: float,
        seed: int, batch: int, output_dir: Path, filename: str,
    ) -> ToolResult | None:
        api_key = config.cloud_image_gen.replicate_api_key
        if not api_key:
            return None

        model = config.cloud_image_gen.replicate_model
        url = "https://api.replicate.com/v1/predictions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "version": model.split(":")[-1] if ":" in model else model,
            "input": {
                "prompt": prompt,
                "negative_prompt": negative or "low quality, blurry",
                "width": width,
                "height": height,
                "num_inference_steps": steps,
                "guidance_scale": cfg,
                "seed": seed,
                "num_outputs": batch,
            },
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(PROVIDER_TIMEOUT)) as client:
            # 提交
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            prediction_id = data.get("id")
            if not prediction_id:
                return None

            # 轮询直到完成
            deadline = time.time() + 300.0
            while time.time() < deadline:
                poll_resp = await client.get(f"{url}/{prediction_id}", headers=headers)
                poll_resp.raise_for_status()
                poll_data = poll_resp.json()

                status = poll_data.get("status")
                if status == "succeeded":
                    output = poll_data.get("output", [])
                    image_urls = output if isinstance(output, list) else [output]
                    images = await self._download_images(image_urls, client)
                    saved = self._save_images(images, output_dir, filename, batch)
                    return ToolResult.ok(
                        data={
                            "images": saved,
                            "seed": seed,
                            "width": width,
                            "height": height,
                            "gen_method": "replicate",
                        },
                        metadata={"prompt": prompt[:200], "seed": seed, "provider": "replicate"},
                    )
                elif status == "failed":
                    log.warn(f"Replicate 预测失败: {poll_data.get('error')}")
                    return None

                await asyncio.sleep(2.0)

            return None

    # ── Modal ─────────────────────────────────────────

    async def _call_modal(
        self, prompt: str, negative: str,
        width: int, height: int, steps: int, cfg: float,
        seed: int, batch: int, output_dir: Path, filename: str,
    ) -> ToolResult | None:
        endpoint = config.cloud_image_gen.modal_endpoint
        api_key = config.cloud_image_gen.modal_api_key
        if not endpoint:
            return None

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        body = {
            "prompt": prompt,
            "negative_prompt": negative or "low quality, blurry",
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg,
            "seed": seed,
            "num_images": batch,
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(PROVIDER_TIMEOUT)) as client:
            resp = await client.post(endpoint, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            image_urls = data.get("images", [])
            if isinstance(data, list):
                image_urls = data
            if isinstance(data, str):
                image_urls = [data]

            if not image_urls:
                return None

            images = await self._download_images(image_urls, client)
            saved = self._save_images(images, output_dir, filename, batch)

            return ToolResult.ok(
                data={
                    "images": saved,
                    "seed": seed,
                    "width": width,
                    "height": height,
                    "gen_method": "modal",
                },
                metadata={"prompt": prompt[:200], "seed": seed, "provider": "modal"},
            )

    # ── Workflow builder (LoRA aware) ─────────────────

    # SDXL quality tags prepended to every positive prompt
    QUALITY_PREFIX = "masterpiece, best quality, highly detailed, sharp focus, professional lighting, "
    # Comprehensive negative prompt for SDXL
    QUALITY_NEGATIVE = (
        "low quality, blurry, distorted, ugly, bad anatomy, bad hands, missing fingers, "
        "extra fingers, fused fingers, poorly drawn face, poorly drawn hands, "
        "watermark, text, logo, signature, jpeg artifacts, grain, noise, "
        "oversaturated, overexposed, underexposed, bad composition, deformed"
    )

    def _build_comfyui_workflow(
        self, prompt: str, negative: str,
        width: int, height: int, steps: int, cfg: float,
        seed: int, batch: int,
        lora_path: str = "", trigger_word: str = "",
    ) -> dict:
        """Build ComfyUI workflow — direct SDXL Turbo generation.

        Pipeline: txt2img → VAE decode → SaveImage
        LoRA injected between checkpoint and KSampler when lora_path is set.
        SDXL Turbo generates high-quality images in 4-8 steps, no upscale needed.
        """
        checkpoint = config.image_gen.sdxl_model

        # 品质前缀 + trigger word
        effective_prompt = self.QUALITY_PREFIX + prompt
        if trigger_word and trigger_word not in effective_prompt:
            effective_prompt = f"{trigger_word}, {effective_prompt}"

        effective_negative = negative or self.QUALITY_NEGATIVE

        workflow = {
            "1": {
                "inputs": {"ckpt_name": checkpoint},
                "class_type": "CheckpointLoaderSimple",
            },
            "2": {
                "inputs": {"text": effective_prompt, "clip": ["1", 1]},
                "class_type": "CLIPTextEncode",
            },
            "3": {
                "inputs": {"text": effective_negative, "clip": ["1", 1]},
                "class_type": "CLIPTextEncode",
            },
        }

        node_idx = 4

        # ── Optional: LoRA injection ──
        if lora_path:
            lora_name = Path(lora_path).name
            workflow[str(node_idx)] = {
                "inputs": {
                    "lora_name": lora_name,
                    "strength_model": 1.0,
                    "strength_clip": 1.0,
                    "model": ["1", 0],
                    "clip": ["1", 1],
                },
                "class_type": "LoraLoader",
            }
            lora_node = str(node_idx)
            model_ref = [lora_node, 0]
            clip_ref = [lora_node, 1]
            node_idx += 1
        else:
            model_ref = ["1", 0]
            clip_ref = ["1", 1]

        vae_ref = ["1", 2]

        # ── Core generation nodes: direct txt2img, no upscale ──
        lat_node = str(node_idx)
        ksampler_node = str(node_idx + 1)
        vae_node = str(node_idx + 2)
        save_node = str(node_idx + 3)

        workflow[lat_node] = {
            "inputs": {"width": width, "height": height, "batch_size": batch},
            "class_type": "EmptyLatentImage",
        }
        workflow[ksampler_node] = {
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "dpmpp_sde",
                "scheduler": "karras",
                "denoise": 1.0,
                "model": model_ref,
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": [lat_node, 0],
            },
            "class_type": "KSampler",
        }
        workflow[vae_node] = {
            "inputs": {"samples": [ksampler_node, 0], "vae": vae_ref},
            "class_type": "VAEDecode",
        }
        workflow[save_node] = {
            "inputs": {"filename_prefix": "df_gen", "images": [vae_node, 0]},
            "class_type": "SaveImage",
        }

        return workflow

    # ── Helpers ───────────────────────────────────────

    async def _download_images(self, urls: list[str], client: httpx.AsyncClient) -> list[bytes]:
        images = []
        for url in urls:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                images.append(resp.content)
            except Exception as e:
                log.warn(f"下载图像失败: {url} — {e}")
        return images

    def _save_images(self, images: list[bytes], output_dir: Path, filename: str, batch: int) -> list[str]:
        saved = []
        for i, img_data in enumerate(images):
            name = filename or f"{uuid.uuid4().hex[:8]}.png"
            if batch > 1:
                name = f"{Path(name).stem}_{i}{Path(name).suffix}"
            path = output_dir / name
            path.write_bytes(img_data)
            saved.append(str(path))
        return saved

    # ── Mock ──────────────────────────────────────────

    def _mock_generate(
        self, prompt: str, negative: str,
        width: int, height: int,
        output_dir: Path, filename: str, seed: int,
    ) -> ToolResult:
        name = filename or f"{uuid.uuid4().hex[:8]}.png"
        path = output_dir / name
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            from PIL import Image
            img = Image.new("RGB", (width, height), color=(40, 40, 45))
            img.save(str(path))
        except Exception:
            path.write_bytes(b"")

        return ToolResult.ok(
            data={
                "images": [str(path)],
                "seed": seed if seed != -1 else hash(prompt) & 0x7FFFFFFF,
                "width": width,
                "height": height,
                "gen_method": "mock",
            },
            metadata={"prompt": prompt[:200], "mock": True, "provider": "mock"},
        )

    async def close(self):
        if self._http:
            await self._http.aclose()
            self._http = None
