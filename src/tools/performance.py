"""性能/调度类工具 — Prompt 缓存、GPU 资源调度、检查点管理"""

from __future__ import annotations

import hashlib
import time
import json
from pathlib import Path
from typing import Any
from src.tools.base import BaseTool, ToolCall, ToolResult
from src.tools.asset_db import asset_db


class PromptCacheTool(BaseTool):
    """
    Prompt 缓存工具 — 存储已验证的 prompt 模板到 prompt_repo

    策略:
      - 按 shot_spec 特征哈希索引 (framing + scene_type + emotion + characters)
      - 质量评分 ≥ 0.85 的 prompt 自动缓存
      - 相似镜头直接复用已验证 prompt，减少 LLM 调用
    """

    def __init__(self):
        super().__init__("prompt_cache")
        self._hit_count = 0
        self._miss_count = 0

    def schema(self) -> dict:
        return {
            "name": "prompt_cache",
            "description": "Cache and retrieve verified Stable Diffusion prompts",
            "parameters": {
                "shot_spec": {"type": "object"},
                "prompt": {"type": "string"},
                "negative_prompt": {"type": "string"},
                "params": {"type": "object"},
                "quality_score": {"type": "number"},
            },
        }

    async def execute(self, call: ToolCall) -> ToolResult:
        shot_spec = call.params.get("shot_spec", {})
        prompt = call.params.get("prompt", "")
        negative = call.params.get("negative_prompt", "")
        params = call.params.get("params", {})
        quality_score = call.params.get("quality_score", 0)

        # 存储模式 (有 prompt 且质量达标)
        if prompt and quality_score >= 0.85:
            return await self._store(shot_spec, prompt, negative, params, quality_score)

        # 检索模式
        return await self._retrieve(shot_spec)

    async def _store(self, shot_spec: dict, prompt: str, negative: str, params: dict, score: float) -> ToolResult:
        spec_hash = self._hash_spec(shot_spec)
        key = f"prompt:{spec_hash}"

        cache_entry = {
            "prompt": prompt,
            "negative_prompt": negative,
            "params": params,
            "quality_score": score,
            "shot_spec_summary": {
                "framing": shot_spec.get("framing", ""),
                "emotion": shot_spec.get("emotion", ""),
                "scene_id": shot_spec.get("scene_id", ""),
            },
            "cached_at": time.time(),
            "use_count": 1,
        }

        # 更新或创建
        existing = asset_db.get("prompt_repo", key)
        if existing and existing.get("data", {}).get("quality_score", 0) < score:
            cache_entry["use_count"] = existing["data"].get("use_count", 0) + 1
        elif existing:
            cache_entry["use_count"] = existing["data"].get("use_count", 0)
            # 已有的质量更高，保持不变
            return ToolResult.ok(data={"cached": False, "reason": "existing has higher quality", "key": key})

        asset_db.put("prompt_repo", key, cache_entry, {"type": "sd_prompt", "hash": spec_hash})
        asset_db.lock("prompt_repo", key)

        return ToolResult.ok(data={"cached": True, "key": key, "hash": spec_hash})

    async def _retrieve(self, shot_spec: dict) -> ToolResult:
        spec_hash = self._hash_spec(shot_spec)

        # 精确匹配
        exact = asset_db.get("prompt_repo", f"prompt:{spec_hash}")
        if exact:
            self._hit_count += 1
            data = exact["data"]
            # 增加使用计数
            data["use_count"] = data.get("use_count", 0) + 1
            asset_db.put("prompt_repo", f"prompt:{spec_hash}", data, {"type": "sd_prompt"})
            return ToolResult.ok(data={
                "hit": True, "match_type": "exact",
                "prompt": data["prompt"],
                "negative_prompt": data.get("negative_prompt", ""),
                "params": data.get("params", {}),
                "quality_score": data.get("quality_score", 0),
                "use_count": data["use_count"],
            })

        # 相似搜索
        search_query = f"{shot_spec.get('emotion', '')} {shot_spec.get('framing', '')} {shot_spec.get('action', '')}"
        similar = asset_db.search("prompt_repo", search_query, top_k=3)
        if similar and similar[0]["score"] > 1.0:
            self._hit_count += 1
            best = similar[0]
            return ToolResult.ok(data={
                "hit": True, "match_type": "similar",
                "score": best["score"],
                "prompt": best["data"].get("prompt", ""),
                "negative_prompt": best["data"].get("negative_prompt", ""),
                "params": best["data"].get("params", {}),
                "quality_score": best["data"].get("quality_score", 0),
                "similar_matches": similar,
            })

        self._miss_count += 1
        return ToolResult.ok(data={
            "hit": False, "match_type": "none",
            "suggestion": "generate_new",
            "spec_hash": spec_hash,
        })

    def _hash_spec(self, shot_spec: dict) -> str:
        """生成 shot spec 的确定性哈希"""
        canonical = json.dumps({
            "framing": shot_spec.get("framing", ""),
            "emotion": shot_spec.get("emotion", ""),
            "movement": shot_spec.get("camera_movement", ""),
            "scene_id": shot_spec.get("scene_id", ""),
            "action": shot_spec.get("action_description", "")[:100],
            "chars": sorted(shot_spec.get("characters_in_frame", [])),
        }, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:20]

    def get_stats(self) -> dict:
        total = self._hit_count + self._miss_count
        hit_rate = self._hit_count / max(total, 1) * 100
        return {
            "hits": self._hit_count,
            "misses": self._miss_count,
            "total_lookups": total,
            "hit_rate_pct": round(hit_rate, 1),
            "cache_entries": len(asset_db._store.get("prompt_repo", {})),
        }


class GPUSchedulerTool(BaseTool):
    """
    GPU 资源调度工具 — 管理虚拟 GPU 资源的分配与释放

    生产环境: 对接 Kubernetes / Slurm GPU 集群
    MVP: 模拟 VRAM 分配，追踪并发任务
    """

    def __init__(self, total_vram_gb: float = 24.0, max_concurrent: int = 8):
        super().__init__("gpu_scheduler")
        self._total_vram = total_vram_gb
        self._available_vram = total_vram_gb
        self._max_concurrent = max_concurrent
        self._active_jobs: dict[str, dict] = {}
        self._job_queue: list[dict] = []
        self._job_counter = 0

    def schema(self) -> dict:
        return {
            "name": "gpu_scheduler",
            "description": "Allocate and release GPU resources for image generation jobs",
            "parameters": {
                "action": {"type": "string"},
                "job_id": {"type": "string"},
                "vram_required_gb": {"type": "number", "default": 2.5},
                "priority": {"type": "integer", "default": 0},
            },
        }

    async def execute(self, call: ToolCall) -> ToolResult:
        action = call.params.get("action", "request")

        if action == "request":
            return self._request_vram(call.params)
        elif action == "release":
            return self._release_vram(call.params)
        elif action == "status":
            return self._get_status()
        elif action == "set_limits":
            return self._set_limits(call.params)
        else:
            return ToolResult.fail(data=None, suggestions=[f"Unknown action: {action}"])

    def _request_vram(self, params: dict) -> ToolResult:
        vram = params.get("vram_required_gb", 2.5)
        priority = params.get("priority", 0)

        self._job_counter += 1
        job_id = params.get("job_id", f"job_{self._job_counter}")
        job = {
            "job_id": job_id,
            "vram_gb": vram,
            "priority": priority,
            "queued_at": time.time(),
        }

        # 检查资源是否可用
        if len(self._active_jobs) >= self._max_concurrent or self._available_vram < vram:
            self._job_queue.append(job)
            self._job_queue.sort(key=lambda j: j["priority"], reverse=True)
            return ToolResult.ok(data={
                "allocated": False,
                "job_id": job_id,
                "status": "queued",
                "queue_position": len(self._job_queue),
                "available_vram_gb": self._available_vram,
                "active_jobs": len(self._active_jobs),
            })

        self._active_jobs[job_id] = job
        self._available_vram -= vram

        return ToolResult.ok(data={
            "allocated": True,
            "job_id": job_id,
            "status": "running",
            "vram_allocated_gb": vram,
            "remaining_vram_gb": round(self._available_vram, 1),
            "active_jobs": len(self._active_jobs),
        })

    def _release_vram(self, params: dict) -> ToolResult:
        job_id = params.get("job_id", "")
        if job_id in self._active_jobs:
            job = self._active_jobs.pop(job_id)
            self._available_vram += job["vram_gb"]
            completed = {"job_id": job_id, "vram_released_gb": job["vram_gb"]}

            # 尝试从队列中取下一个
            started = []
            while self._job_queue and len(self._active_jobs) < self._max_concurrent:
                next_job = self._job_queue.pop(0)
                if self._available_vram >= next_job["vram_gb"]:
                    self._active_jobs[next_job["job_id"]] = next_job
                    self._available_vram -= next_job["vram_gb"]
                    started.append(next_job["job_id"])

            return ToolResult.ok(data={
                "released": completed,
                "started_from_queue": started,
                "remaining_vram_gb": round(self._available_vram, 1),
                "active_jobs": len(self._active_jobs),
            })
        return ToolResult.ok(data={"released": None, "note": "job not found in active jobs"})

    def _get_status(self) -> ToolResult:
        return ToolResult.ok(data={
            "total_vram_gb": self._total_vram,
            "available_vram_gb": round(self._available_vram, 1),
            "used_vram_gb": round(self._total_vram - self._available_vram, 1),
            "utilization_pct": round((1 - self._available_vram / self._total_vram) * 100, 1),
            "active_jobs": len(self._active_jobs),
            "max_concurrent": self._max_concurrent,
            "queue_length": len(self._job_queue),
            "active_job_ids": list(self._active_jobs.keys()),
        })

    def _set_limits(self, params: dict) -> ToolResult:
        if "total_vram_gb" in params:
            self._total_vram = params["total_vram_gb"]
        if "max_concurrent" in params:
            self._max_concurrent = params["max_concurrent"]
        return ToolResult.ok(data={
            "total_vram_gb": self._total_vram,
            "max_concurrent": self._max_concurrent,
        })


class CheckpointTool(BaseTool):
    """
    管线检查点工具 — 保存/恢复管线状态

    支持:
      - 完整管线状态序列化到 JSON
      - 从检查点恢复 (跳过已完成的 Phase)
      - 增量重试 (仅重跑失败的镜头)
    """

    def __init__(self, default_dir: str = "outputs/checkpoints"):
        super().__init__("checkpoint")
        self._default_dir = Path(default_dir)
        self._default_dir.mkdir(parents=True, exist_ok=True)

    def schema(self) -> dict:
        return {
            "name": "checkpoint",
            "description": "Save or restore pipeline execution state",
            "parameters": {
                "action": {"type": "string"},
                "project_id": {"type": "string"},
                "state": {"type": "object"},
                "checkpoint_path": {"type": "string"},
            },
        }

    async def execute(self, call: ToolCall) -> ToolResult:
        action = call.params.get("action", "save")
        project_id = call.params.get("project_id", f"project_{int(time.time())}")

        if action == "save":
            return self._save(project_id, call.params.get("state", {}))
        elif action == "load":
            return self._load(project_id)
        elif action == "list":
            return self._list_checkpoints()
        elif action == "diff":
            return self._diff(call.params)
        else:
            return ToolResult.fail(data=None, suggestions=[f"Unknown action: {action}"])

    def _save(self, project_id: str, state: dict) -> ToolResult:
        checkpoint = {
            "project_id": project_id,
            "saved_at": time.time(),
            "saved_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "state": state,
            "version": state.get("version", 1),
        }

        cp_path = self._default_dir / f"{project_id}.json"
        cp_path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")

        return ToolResult.ok(data={
            "project_id": project_id,
            "checkpoint_path": str(cp_path),
            "saved_at": checkpoint["saved_at_iso"],
            "state_keys": list(state.keys()),
        })

    def _load(self, project_id: str) -> ToolResult:
        cp_path = self._default_dir / f"{project_id}.json"
        if not cp_path.exists():
            return ToolResult.ok(data={"found": False, "project_id": project_id})

        checkpoint = json.loads(cp_path.read_text(encoding="utf-8"))
        state = checkpoint.get("state", {})

        # 分析哪些已完成
        completed_phases = []
        if state.get("screenplay"):
            completed_phases.append("creative")
        if state.get("char_profiles"):
            completed_phases.append("character")
        if state.get("completed_shots"):
            completed_phases.append("shots")

        return ToolResult.ok(data={
            "found": True,
            "project_id": project_id,
            "saved_at": checkpoint.get("saved_at_iso", ""),
            "state": state,
            "completed_phases": completed_phases,
            "completed_shot_count": len(state.get("completed_shots", [])),
            "failed_shot_count": len(state.get("failed_shots", [])),
        })

    def _list_checkpoints(self) -> ToolResult:
        checkpoints = []
        for f in sorted(self._default_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                checkpoints.append({
                    "project_id": data.get("project_id", f.stem),
                    "saved_at": data.get("saved_at_iso", ""),
                    "version": data.get("version", 1),
                })
            except Exception:
                pass
        return ToolResult.ok(data={"checkpoints": checkpoints, "count": len(checkpoints)})

    def _diff(self, params: dict) -> ToolResult:
        """比较两个检查点或当前状态与检查点"""
        cp1 = params.get("checkpoint_1", "")
        cp2 = params.get("checkpoint_2", "")

        p1 = self._default_dir / f"{cp1}.json"
        p2 = self._default_dir / f"{cp2}.json"

        if not p1.exists():
            return ToolResult.ok(data={"diff": {}, "note": f"checkpoint {cp1} not found"})

        s1 = json.loads(p1.read_text(encoding="utf-8")).get("state", {})
        s2 = json.loads(p2.read_text(encoding="utf-8")).get("state", {}) if p2.exists() else {}

        shots1 = len(s1.get("completed_shots", []))
        shots2 = len(s2.get("completed_shots", []))

        return ToolResult.ok(data={
            "checkpoint_1_shots": shots1,
            "checkpoint_2_shots": shots2,
            "delta_shots": shots1 - shots2,
        })
