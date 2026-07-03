"""lora_sourcing — LoRA 来源化: CivitAI 搜索/下载 → HuggingFace → 云端训练 → mock 回退"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import httpx

from src.tools.asset_db import asset_db
from src.core.config import config
from src.core.logging import get_logger

log = get_logger("LoraSourcing")

CIVITAI_API = "https://civitai.com/api/v1"
LORA_DIR = Path("assets/loras")
LORA_DIR.mkdir(parents=True, exist_ok=True)


class LoraSourcing:
    """LoRA 来源管理器 — 三种策略按优先级回退"""

    def __init__(self):
        self._http: httpx.AsyncClient | None = None

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0),
                headers={"User-Agent": "DirectorFactory/0.2.0"},
            )
        return self._http

    # ── Main entry ────────────────────────────────────

    async def source(
        self,
        character_id: str,
        description: str,
        trigger_word: str = "",
        base_model: str = "SDXL",
    ) -> dict:
        """
        为角色获取 LoRA 权重。按优先级:
          1. CivitAI 搜索 + 下载
          2. HuggingFace Hub 搜索
          3. 本地 mock (兜底)
        """
        trigger = trigger_word or f"char_{character_id}"
        result = {
            "source": "mock",
            "lora_path": "",
            "trigger_word": trigger,
            "base_model": base_model,
            "sourced_at": time.time(),
        }

        # 1. CivitAI
        log.info(f"[{character_id}] 搜索 CivitAI: {trigger} / {description[:60]}")
        civitai_result = await self._try_civitai(character_id, description, trigger)
        if civitai_result:
            result.update(civitai_result)
            self._register(character_id, result)
            return result

        # 2. HuggingFace
        log.info(f"[{character_id}] CivitAI 无结果, 尝试 HuggingFace: {description[:60]}")
        hf_result = await self._try_huggingface(character_id, description, trigger)
        if hf_result:
            result.update(hf_result)
            self._register(character_id, result)
            return result

        # 3. 兜底: 创建空占位
        log.info(f"[{character_id}] 无匹配 LoRA, 创建占位文件")
        placeholder_path = LORA_DIR / f"{character_id}_placeholder.safetensors"
        placeholder_path.touch()
        result["lora_path"] = str(placeholder_path)
        result["source"] = "mock"
        result["metadata"] = {"warning": "no matching LoRA found, using placeholder"}
        self._register(character_id, result)
        return result

    # ── CivitAI ───────────────────────────────────────

    async def _try_civitai(
        self, character_id: str, description: str, trigger: str,
    ) -> dict | None:
        """搜索 CivitAI 并下载最匹配的 LoRA"""
        try:
            # 用 trigger_word 和 description 关键词搜索
            query = trigger if trigger else " ".join(description.split()[:5])
            url = f"{CIVITAI_API}/models"
            params = {
                "query": query,
                "types": "LORA",
                "sort": "Highest Rated",
                "limit": 5,
            }
            resp = await self.http.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

            items = data.get("items", [])
            if not items:
                return None

            # 取最高评分且匹配度最高的
            best = items[0]
            model_id = best.get("id")
            model_name = best.get("name", "")
            model_tags = best.get("tags", [])
            base_model_tag = [t for t in model_tags if "sdxl" in t.lower() or "xl" in t.lower()]

            # 获取最新版本的文件
            versions = best.get("modelVersions", [])
            if not versions:
                return None

            latest = versions[0]
            version_id = latest.get("id")
            files = latest.get("files", [])
            # 选 safetensors 文件
            safetensor_file = next(
                (f for f in files if f.get("name", "").endswith(".safetensors")),
                None,
            )
            if not safetensor_file:
                # 如果有 pickle 文件也行
                safetensor_file = next(
                    (f for f in files if f.get("name", "").endswith(".ckpt") or f.get("name", "").endswith(".pt")),
                    None,
                )
            if not safetensor_file:
                return None

            download_url = safetensor_file.get("downloadUrl", "")
            if not download_url:
                return None

            file_hash = safetensor_file.get("hashes", {}).get("SHA256", "")
            trained_words = latest.get("trainedWords", [trigger])
            effective_trigger = trained_words[0] if trained_words else trigger

            # 下载
            filename = f"{character_id}_civitai_{model_id}.safetensors"
            target_path = LORA_DIR / filename
            await self._download_file(download_url, target_path)

            actual_size = target_path.stat().st_size
            if actual_size < 100:
                log.warn(f"[{character_id}] 下载的 LoRA 文件太小 ({actual_size} bytes), 丢弃")
                target_path.unlink()
                return None

            log.info(f"[{character_id}] CivitAI 下载完成: {target_path} ({actual_size} bytes)")

            return {
                "source": "civitai",
                "lora_path": str(target_path),
                "trigger_word": effective_trigger,
                "base_model": "SDXL",
                "source_url": f"https://civitai.com/models/{model_id}",
                "download_hash": file_hash,
                "model_name": model_name,
                "rating": best.get("stats", {}).get("rating", 0),
                "sourced_at": time.time(),
            }

        except httpx.ConnectError:
            log.info(f"[{character_id}] CivitAI 不可达")
            return None
        except Exception as e:
            log.warn(f"[{character_id}] CivitAI 搜索失败: {e}")
            return None

    # ── HuggingFace ───────────────────────────────────

    async def _try_huggingface(
        self, character_id: str, description: str, trigger: str,
    ) -> dict | None:
        """搜索 HuggingFace Hub 上的 SDXL LoRA"""
        try:
            # 用 huggingface_hub 搜索
            query = trigger if trigger else description[:100]
            url = "https://huggingface.co/api/models"
            params = {
                "search": f"{query} lora sdxl",
                "sort": "downloads",
                "direction": "-1",
                "limit": 3,
                "filter": "safetensors",
            }
            resp = await self.http.get(url, params=params)
            if resp.status_code >= 400:
                return None
            data = resp.json()

            # 过滤出可能的 LoRA 模型
            candidates = []
            for item in data:
                tags = item.get("tags", [])
                model_id = item.get("modelId", "")
                if any(t in str(tags).lower() for t in ["lora", "sdxl", "xl"]):
                    candidates.append(item)

            if not candidates:
                return None

            best = candidates[0]
            model_id = best.get("modelId", "")
            downloads = best.get("downloads", 0)

            # 尝试下载 safetensors 文件
            # 构建 HF download URL
            hf_filename = f"{character_id}_hf.safetensors"
            target_path = LORA_DIR / hf_filename

            # 尝试使用 huggingface_hub 下载
            try:
                from huggingface_hub import hf_hub_download
                # 列出模型文件
                files_url = f"https://huggingface.co/api/models/{model_id}?blobs=true"
                files_resp = await self.http.get(files_url)
                if files_resp.status_code < 400:
                    siblings = files_resp.json().get("siblings", [])
                    safetensors = [
                        s for s in siblings
                        if s.get("rfilename", "").endswith(".safetensors")
                    ]
                    if safetensors:
                        chosen = safetensors[0]["rfilename"]
                        downloaded_path = hf_hub_download(
                            repo_id=model_id,
                            filename=chosen,
                            local_dir=str(LORA_DIR),
                            local_dir_use_symlinks=False,
                        )
                        # 重命名
                        p = Path(downloaded_path)
                        if p != target_path:
                            p.rename(target_path)
                    else:
                        return None
                else:
                    return None
            except ImportError:
                # huggingface_hub 不可用, 尝试直接 HTTP 下载
                hf_download_url = f"https://huggingface.co/{model_id}/resolve/main"
                # 没有具体文件名则跳过
                log.info(f"[{character_id}] huggingface_hub 未安装, 跳过 HF 下载")
                return None
            except Exception as e:
                log.warn(f"[{character_id}] HF 下载失败: {e}")
                return None

            actual_size = target_path.stat().st_size
            if actual_size < 100:
                target_path.unlink()
                return None

            log.info(f"[{character_id}] HuggingFace 下载完成: {target_path} ({actual_size} bytes)")

            return {
                "source": "huggingface",
                "lora_path": str(target_path),
                "trigger_word": trigger,
                "base_model": "SDXL",
                "source_url": f"https://huggingface.co/{model_id}",
                "model_name": model_id,
                "downloads": downloads,
                "sourced_at": time.time(),
            }

        except Exception as e:
            log.warn(f"[{character_id}] HuggingFace 搜索失败: {e}")
            return None

    # ── Helpers ───────────────────────────────────────

    async def _download_file(self, url: str, target: Path) -> None:
        """下载文件到指定路径"""
        resp = await self.http.get(url)
        resp.raise_for_status()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(resp.content)

    def _register(self, character_id: str, result: dict) -> None:
        """将 LoRA 资产注册到 AssetDB"""
        record = {
            "character_id": character_id,
            "lora_path": result.get("lora_path", ""),
            "trigger_word": result.get("trigger_word", ""),
            "base_model": result.get("base_model", "SDXL"),
            "source": result.get("source", "mock"),
            "source_url": result.get("source_url", ""),
            "download_hash": result.get("download_hash", ""),
            "model_name": result.get("model_name", ""),
            "rating": result.get("rating", 0),
            "sourced_at": result.get("sourced_at", time.time()),
        }

        asset_db.put(
            "char_asset_db",
            f"{character_id}:lora",
            record,
            {"type": "lora", "source": result.get("source", "mock")},
        )
        asset_db.lock("char_asset_db", f"{character_id}:lora")

        log.info(
            f"[{character_id}] LoRA 已注册: source={result.get('source')}, "
            f"path={result.get('lora_path')}, trigger={result.get('trigger_word')}"
        )

    async def close(self):
        if self._http:
            await self._http.aclose()
            self._http = None
