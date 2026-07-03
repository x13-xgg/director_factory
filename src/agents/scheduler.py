"""制片调度 Agent — 并行策略、资源分配、依赖分析、GPU 调度 + 关键路径 + 动态批处理"""

import time
from collections import defaultdict, deque
from src.core.agent import BaseAgent, AgentConfig
from src.data.protocols import ProductionTask, ShotStatus


class SchedulerAgent(BaseAgent):
    """
    职责:
      1. ShotList → 可并行批次分解 (4 条件并行策略)
      2. GPU 资源感知的批次大小控制 (+ 动态调整)
      3. 依赖图构建 + 关键路径分析 + ETA 预估
      4. 吞吐量与利用率统计 + 单镜头耗时追踪

    并行条件 (按安全等级排序):
      L1. 不同场景 → 安全并行 (场景/光照参数独立)
      L2. 同场景不共享角色 → 安全并行
      L3. 同场景同角色无前后依赖 → 可行 (需总监确认)
      L4. 有依赖关系 → 必须串行
    """

    MAX_BATCH_SIZE = 8  # 单批次最大并行数
    GPU_VRAM_PER_SHOT = 2.5  # 每镜头预估 VRAM (GB)
    GPU_TOTAL_VRAM = 24.0  # 总 VRAM (GB)
    MAX_CONCURRENT_BY_VRAM = int(GPU_TOTAL_VRAM / GPU_VRAM_PER_SHOT)

    def __init__(self, config: AgentConfig, bus, tools):
        super().__init__(config, bus, tools)
        self._shot_status: dict[str, str] = {}
        self._completed: set[str] = set()
        self._tasks: dict[str, ProductionTask] = {}
        self._order: list[str] = []

        # 依赖图与统计
        self._dep_graph: dict[str, set[str]] = {}  # shot_id → {prerequisites}
        self._reverse_deps: dict[str, set[str]] = {}  # shot_id → {dependents}
        self._scene_groups: dict[str, list[str]] = {}  # scene_id → [shot_ids]
        self._batch_history: list[dict] = []
        self._stats = {
            "total_batches": 0,
            "total_shots_completed": 0,
            "avg_batch_size": 0.0,
            "gpu_utilization": 0.0,
            "throughput_per_min": 0.0,
            "start_time": 0.0,
        }

        # 动态调度增强
        self._shot_timings: dict[str, float] = {}  # shot_id → actual duration seconds
        self._avg_shot_duration: float = 30.0  # 初始预估 30s/镜头
        self._critical_path: list[str] = []  # 关键路径 shot_ids
        self._critical_path_duration: float = 0.0
        self._dynamic_batch_limit: int = self.MAX_CONCURRENT_BY_VRAM

    async def handle_task(self, task: dict) -> dict:
        action = task.get("action", "init")

        if action == "init":
            return await self._init(task)
        elif action == "next_batch":
            return await self._next_batch(task)
        elif action == "mark_done":
            return await self._mark_done(task.get("shot_id", ""), task.get("elapsed", 0))
        elif action == "retry":
            return await self._retry(task.get("shot_id", ""), task.get("feedback", ""))
        elif action == "all_done":
            return {"status": "ok", "all_done": len(self._completed) >= len(self._order)}
        elif action == "stats":
            return {"status": "ok", "stats": self._compute_stats()}
        elif action == "dep_graph":
            return {"status": "ok", "dep_graph": self._export_dep_graph()}
        elif action == "critical_path":
            return {"status": "ok", "critical_path": self._compute_critical_path()}
        elif action == "eta":
            return {"status": "ok", "eta": self._estimate_eta()}
        elif action == "pending_count":
            pending = [s for s in self._order if s not in self._completed
                       and self._shot_status.get(s) not in (ShotStatus.RETRY.value, ShotStatus.FAILED.value)]
            return {"status": "ok", "pending": len(pending)}
        elif action == "blocked_shots":
            return {"status": "ok", "blocked_shots": self._get_blocked_shots()}
        elif action == "predictive_next":
            return await self._predictive_next_batch()
        elif action == "metrics":
            return {"status": "ok", "metrics": self._export_prometheus_metrics()}
        else:
            return {"status": "error", "error": f"Unknown action: {action}"}

    async def _init(self, task: dict) -> dict:
        shotlist = task.get("shotlist", {})
        shots = shotlist.get("shots", [])

        self._shot_status.clear()
        self._completed.clear()
        self._tasks.clear()
        self._order.clear()
        self._dep_graph.clear()
        self._reverse_deps.clear()
        self._scene_groups.clear()
        self._batch_history.clear()
        self._shot_timings.clear()
        self._avg_shot_duration = 30.0
        self._critical_path = []
        self._critical_path_duration = 0.0
        self._dynamic_batch_limit = self.MAX_CONCURRENT_BY_VRAM
        self._stats["start_time"] = time.time()

        for s in shots:
            sid = s.get("id", "") if isinstance(s, dict) else s.id
            self._shot_status[sid] = ShotStatus.PENDING.value
            self._tasks[sid] = ProductionTask(
                shot=s,
                retry_count=0,
                max_retries=task.get("max_retries", 3),
            )
            self._order.append(sid)

        # 构建依赖图
        self._build_dependency_graph(shots)
        # 按场景分组
        self._build_scene_groups(shots)
        # 计算关键路径
        cp = self._compute_critical_path()
        self._critical_path = cp["critical_path"]
        self._critical_path_duration = cp["estimated_duration_seconds"]

        self.log.info(
            f"初始化调度: {len(shots)} 镜头, "
            f"{len(self._scene_groups)} 场景, "
            f"{sum(1 for deps in self._dep_graph.values() if deps)} 个有依赖, "
            f"最大并行={self.MAX_CONCURRENT_BY_VRAM} (VRAM={self.GPU_TOTAL_VRAM}GB/{self.GPU_VRAM_PER_SHOT}GB per shot), "
            f"关键路径={len(self._critical_path)}镜头/{self._critical_path_duration:.0f}s"
        )

        return {
            "status": "ok",
            "total_shots": len(shots),
            "order": self._order,
            "scene_groups": {k: len(v) for k, v in self._scene_groups.items()},
            "max_concurrent": self.MAX_CONCURRENT_BY_VRAM,
            "critical_path": self._critical_path,
            "critical_path_duration_s": self._critical_path_duration,
            "estimated_total_duration_s": self._estimate_eta()["estimated_total_seconds"],
        }

    async def _next_batch(self, task: dict = None) -> dict:
        """根据 4 条件并行策略返回可并行批次"""
        # 收集所有可以执行的镜头 (依赖已满足)
        ready = []
        blocked = []
        for sid in self._order:
            if sid in self._completed:
                continue
            status = self._shot_status.get(sid)
            if status in (ShotStatus.RETRY.value, ShotStatus.FAILED.value):
                continue
            if status == ShotStatus.IN_PROGRESS.value:
                continue
            task_obj = self._tasks.get(sid)
            if not task_obj:
                continue

            shot = task_obj.shot if isinstance(task_obj.shot, dict) else {}
            deps = shot.get("dependencies", [])
            valid_deps = [d for d in deps if d in self._tasks]
            if all(d in self._completed for d in valid_deps):
                ready.append(sid)
            else:
                blocked.append({"shot_id": sid, "blocked_by": [d for d in valid_deps if d not in self._completed]})

        if not ready:
            if blocked:
                self.log.info(f"所有待执行镜头被阻塞: {len(blocked)}")
            return {"status": "ok", "batch": [], "shots": [], "message": "no pending shots", "blocked": blocked}

        # 按安全等级排序 (L1 > L2 > L3 > L4)
        scored = [(sid, self._parallel_safety_score(sid)) for sid in ready]
        scored.sort(key=lambda x: x[1], reverse=True)

        # 选择批次: 最大化场景多样性和 GPU 利用率
        batch = []
        used_scenes: set[str] = set()
        used_chars: dict[str, set[str]] = {}  # scene_id → set of chars in batch

        for sid, safety in scored:
            if len(batch) >= self._dynamic_batch_limit:
                break
            if len(batch) >= self.MAX_BATCH_SIZE:
                break

            task_obj = self._tasks[sid]
            shot = task_obj.shot if isinstance(task_obj.shot, dict) else {}
            scene_id = shot.get("scene_id", "")
            chars = set(shot.get("characters_in_frame", []))

            # L1: different scene is always safe
            if scene_id not in used_scenes:
                batch.append(sid)
                used_scenes.add(scene_id)
                used_chars[scene_id] = chars
            # L2: same scene but no character overlap with other shots in SAME scene
            elif not chars & used_chars.get(scene_id, set()):
                batch.append(sid)
                used_chars[scene_id] |= chars
            # L3: same scene, same chars, no direct dependency → acceptable but warn
            elif safety >= 3:
                batch.append(sid)
                used_chars[scene_id] |= chars
                self.log.info(f"L3 并行 (同场景同角色): {sid} scene={scene_id}")
            # L4: has unmet dependency → can't run (shouldn't be in ready)

        # 标记为 in_progress
        for sid in batch:
            self._shot_status[sid] = ShotStatus.IN_PROGRESS.value

        self._batch_history.append({
            "batch_id": len(self._batch_history),
            "shot_ids": batch,
            "size": len(batch),
            "scene_diversity": len(used_scenes),
        })

        self.log.info(
            f"分配批次 #{len(self._batch_history)}: {len(batch)} 镜头, "
            f"{len(used_scenes)} 场景, "
            f"预估 VRAM={len(batch) * self.GPU_VRAM_PER_SHOT:.1f}/{self.GPU_TOTAL_VRAM}GB"
        )

        return {
            "status": "ok",
            "batch": batch,
            "shots": [self._tasks[sid].shot for sid in batch],
            "batch_id": len(self._batch_history) - 1,
            "scene_diversity": len(used_scenes),
            "estimated_vram_gb": len(batch) * self.GPU_VRAM_PER_SHOT,
        }

    async def _mark_done(self, shot_id: str, elapsed: float = 0) -> dict:
        self._completed.add(shot_id)
        self._shot_status[shot_id] = ShotStatus.DONE.value

        # 追踪单镜头耗时用于动态调整
        if elapsed > 0:
            self._shot_timings[shot_id] = elapsed
            # 指数移动平均更新预估
            alpha = 0.3
            self._avg_shot_duration = alpha * elapsed + (1 - alpha) * self._avg_shot_duration
            # 动态调整批次大小: 如果单镜头耗时 < 预估, 可以增加并行度
            if elapsed < self._avg_shot_duration * 0.7:
                self._dynamic_batch_limit = min(self.MAX_BATCH_SIZE, self._dynamic_batch_limit + 1)
            elif elapsed > self._avg_shot_duration * 1.5 and self._dynamic_batch_limit > 3:
                self._dynamic_batch_limit -= 1

        remaining = len(self._order) - len(self._completed)
        total = len(self._order)
        progress = (total - remaining) / max(total, 1) * 100

        # 检查是否解锁了下游
        unblocked = self._reverse_deps.get(shot_id, set())
        newly_ready = [d for d in unblocked
                       if d not in self._completed
                       and self._shot_status.get(d) == ShotStatus.PENDING.value
                       and all(p in self._completed for p in self._dep_graph.get(d, set()))]

        if newly_ready:
            self.log.info(f"镜头 {shot_id} 完成, 解锁: {newly_ready}")

        eta = self._estimate_eta()

        self.log.info(
            f"镜头完成: {shot_id} ({remaining} 剩余, {progress:.0f}%), "
            f"耗时={elapsed:.1f}s, ETA={eta['estimated_remaining_seconds']:.0f}s, "
            f"动态并发={self._dynamic_batch_limit}"
        )
        return {
            "status": "ok", "shot_id": shot_id, "remaining": remaining,
            "progress_pct": round(progress, 1), "unblocked": newly_ready,
            "shot_elapsed_s": elapsed,
            "avg_shot_duration_s": round(self._avg_shot_duration, 1),
            "dynamic_batch_limit": self._dynamic_batch_limit,
            "eta_s": round(eta["estimated_remaining_seconds"], 0),
            "critical_path_remaining": len([s for s in self._critical_path if s not in self._completed]),
        }

    async def _retry(self, shot_id: str, feedback: str) -> dict:
        task = self._tasks.get(shot_id)
        if not task:
            return {"status": "error", "error": f"Unknown shot: {shot_id}"}

        task.retry_count += 1
        task.feedback = feedback

        if task.retry_count >= task.max_retries:
            self._shot_status[shot_id] = ShotStatus.RETRY.value
            self.log.warn(f"镜头 {shot_id} 达到最大重试 ({task.retry_count}/{task.max_retries})")
            return {
                "status": "retry_exhausted", "shot_id": shot_id,
                "retry_count": task.retry_count, "feedback": feedback,
            }
        else:
            self._shot_status[shot_id] = ShotStatus.PENDING.value
            self.log.info(f"镜头 {shot_id} 重试 ({task.retry_count}/{task.max_retries})")
            return {
                "status": "ok", "shot_id": shot_id,
                "retry_count": task.retry_count, "feedback": feedback,
            }

    def _parallel_safety_score(self, shot_id: str) -> int:
        """计算并行安全评分: 4=最安全(L1), 3=L2, 2=L3, 1=L4"""
        task = self._tasks.get(shot_id)
        if not task:
            return 0
        shot = task.shot if isinstance(task.shot, dict) else {}
        scene_id = shot.get("scene_id", "")
        chars = set(shot.get("characters_in_frame", []))
        deps = [d for d in shot.get("dependencies", []) if d in self._tasks]

        # L1: 检查其他进行中的镜头是否在不同场景
        in_progress = [s for s, st in self._shot_status.items()
                       if st == ShotStatus.IN_PROGRESS.value and s != shot_id]
        in_progress_scenes = set()
        in_progress_chars = set()
        for sid in in_progress:
            t = self._tasks.get(sid)
            if t:
                s = t.shot if isinstance(t.shot, dict) else {}
                in_progress_scenes.add(s.get("scene_id", ""))
                in_progress_chars |= set(s.get("characters_in_frame", []))

        if scene_id not in in_progress_scenes:
            return 4  # L1
        if not chars & in_progress_chars:
            return 3  # L2
        if not deps:
            return 2  # L3
        return 1  # L4

    def _build_dependency_graph(self, shots: list):
        """构建依赖图与反向依赖索引"""
        self._dep_graph.clear()
        self._reverse_deps.clear()
        for s in shots:
            sid = s.get("id", "") if isinstance(s, dict) else s.id
            deps = s.get("dependencies", []) if isinstance(s, dict) else s.dependencies
            valid = [d for d in deps if any(
                (s2.get("id", "") if isinstance(s2, dict) else s2.id) == d
                for s2 in shots
            )]
            self._dep_graph[sid] = set(valid)
            for d in valid:
                self._reverse_deps.setdefault(d, set()).add(sid)

    def _build_scene_groups(self, shots: list):
        """按场景分组镜头"""
        self._scene_groups.clear()
        for s in shots:
            sid = s.get("id", "") if isinstance(s, dict) else s.id
            scene_id = s.get("scene_id", "") if isinstance(s, dict) else s.scene_id
            self._scene_groups.setdefault(scene_id, []).append(sid)

    def _get_blocked_shots(self) -> list[dict]:
        """获取所有被阻塞的镜头及原因"""
        blocked = []
        for sid in self._order:
            if sid in self._completed:
                continue
            status = self._shot_status.get(sid)
            if status in (ShotStatus.RETRY.value, ShotStatus.FAILED.value):
                blocked.append({"shot_id": sid, "reason": "retry_exhausted", "status": status})
                continue
            if status == ShotStatus.IN_PROGRESS.value:
                continue
            unmet = [d for d in self._dep_graph.get(sid, set()) if d not in self._completed]
            if unmet:
                blocked.append({"shot_id": sid, "reason": "dependency", "blocked_by": unmet})
        return blocked

    def _compute_stats(self) -> dict:
        """计算调度统计"""
        import time
        elapsed = max(time.time() - self._stats["start_time"], 1)
        completed = len(self._completed)
        total = len(self._order)
        batches = self._batch_history
        avg_batch = sum(b["size"] for b in batches) / max(len(batches), 1)
        throughput = completed / (elapsed / 60)

        return {
            "total_shots": total,
            "completed": completed,
            "progress_pct": round(completed / max(total, 1) * 100, 1),
            "total_batches": len(batches),
            "avg_batch_size": round(avg_batch, 1),
            "max_batch_size": max((b["size"] for b in batches), default=0),
            "gpu_utilization": round(avg_batch / self.MAX_CONCURRENT_BY_VRAM * 100, 1),
            "throughput_per_min": round(throughput, 1),
            "retries_exhausted": sum(
                1 for s in self._shot_status if self._shot_status[s] == ShotStatus.RETRY.value
            ),
            "blocked_currently": len(self._get_blocked_shots()),
        }

    def _export_dep_graph(self) -> dict:
        return {
            "dependencies": {k: list(v) for k, v in self._dep_graph.items()},
            "reverse_deps": {k: list(v) for k, v in self._reverse_deps.items()},
            "scene_groups": self._scene_groups,
            "order": self._order,
        }

    # ── 关键路径分析 ────────────────────────────────

    def _compute_critical_path(self) -> dict:
        """拓扑排序 + 最长路径计算关键路径。

        对每个节点计算 earliest_start = max(所有前驱的 earliest_finish)。
        关键路径 = 从入度为 0 到出度为 0 的最长路径上的节点集合。
        """
        if not self._dep_graph:
            return {"critical_path": [], "estimated_duration_seconds": 0.0}

        # 拓扑排序 (Kahn)
        in_degree: dict[str, int] = {sid: len(deps) for sid, deps in self._dep_graph.items()}
        queue = deque([sid for sid, deg in in_degree.items() if deg == 0])

        earliest_finish: dict[str, float] = {}
        prev_node: dict[str, str | None] = {}

        while queue:
            node = queue.popleft()
            # earliest_finish[current] = max(所有前驱的 earliest_finish) + 预估时长
            pred_finishes = [
                earliest_finish[p] for p in self._dep_graph.get(node, set())
                if p in earliest_finish
            ]
            start_time = max(pred_finishes) if pred_finishes else 0.0
            # 使用实际耗时或平均预估
            duration = self._shot_timings.get(node, self._avg_shot_duration)
            earliest_finish[node] = start_time + duration

            # 记录最晚前驱
            if pred_finishes:
                max_val = max(pred_finishes)
                for p in self._dep_graph.get(node, set()):
                    if p in earliest_finish and earliest_finish[p] == max_val:
                        prev_node[node] = p
                        break
            else:
                prev_node[node] = None

            # 处理后继
            for dep in self._reverse_deps.get(node, set()):
                if dep in in_degree:
                    in_degree[dep] -= 1
                    if in_degree[dep] == 0:
                        queue.append(dep)

        # 找到最晚结束的节点
        if not earliest_finish:
            return {"critical_path": [], "estimated_duration_seconds": 0.0}

        end_node = max(earliest_finish, key=earliest_finish.get)
        total_duration = earliest_finish[end_node]

        # 回溯构建关键路径
        path = []
        current = end_node
        while current is not None:
            path.append(current)
            current = prev_node.get(current)
        path.reverse()

        return {
            "critical_path": path,
            "estimated_duration_seconds": round(total_duration, 1),
            "path_length": len(path),
            "bottleneck_scenes": list(set(
                self._tasks[sid].shot.get("scene_id", "") if isinstance(self._tasks[sid].shot, dict) else ""
                for sid in path if sid in self._tasks
            )),
        }

    # ── ETA 预估 ────────────────────────────────────

    def _estimate_eta(self) -> dict:
        """基于当前吞吐率和剩余镜头数预估完成时间。"""
        completed = len(self._completed)
        total = len(self._order)
        remaining = total - completed

        if remaining == 0:
            return {
                "estimated_remaining_seconds": 0,
                "estimated_total_seconds": round(time.time() - self._stats["start_time"], 1),
                "confidence": 1.0,
            }

        elapsed = max(time.time() - self._stats["start_time"], 1)
        throughput = completed / (elapsed / 60)  # shots per minute

        if throughput > 0:
            remaining_minutes = remaining / throughput
            remaining_seconds = remaining_minutes * 60
        else:
            # 没有足够数据, 使用平均镜头时长估算
            remaining_seconds = remaining * self._avg_shot_duration / max(self._dynamic_batch_limit, 1)

        # 置信度: 完成的镜头越多越准确
        confidence = min(completed / max(total, 1) * 2, 1.0)

        return {
            "estimated_remaining_seconds": round(remaining_seconds, 1),
            "estimated_total_seconds": round(elapsed + remaining_seconds, 1),
            "current_throughput_per_min": round(throughput, 2),
            "avg_shot_duration_s": round(self._avg_shot_duration, 1),
            "dynamic_batch_limit": self._dynamic_batch_limit,
            "critical_path_remaining": len([s for s in self._critical_path if s not in self._completed]),
            "confidence": round(confidence, 2),
        }

    # ── 预测性预取批次 ──────────────────────────────

    async def _predictive_next_batch(self) -> dict:
        """预计算下一批次 (不实际标记 in_progress), 供管线提前准备资源。

        返回如果现在调用 next_batch 会得到什么, 但不改变状态。
        """
        ready = []
        for sid in self._order:
            if sid in self._completed:
                continue
            status = self._shot_status.get(sid)
            if status in (ShotStatus.RETRY.value, ShotStatus.FAILED.value, ShotStatus.IN_PROGRESS.value):
                continue
            task_obj = self._tasks.get(sid)
            if not task_obj:
                continue
            shot = task_obj.shot if isinstance(task_obj.shot, dict) else {}
            deps = shot.get("dependencies", [])
            valid_deps = [d for d in deps if d in self._tasks]
            if all(d in self._completed for d in valid_deps):
                ready.append(sid)

        scored = [(sid, self._parallel_safety_score(sid)) for sid in ready]
        scored.sort(key=lambda x: x[1], reverse=True)

        predicted = []
        used_scenes: set[str] = set()
        used_chars: dict[str, set[str]] = {}

        for sid, safety in scored:
            if len(predicted) >= self._dynamic_batch_limit:
                break
            task_obj = self._tasks[sid]
            shot = task_obj.shot if isinstance(task_obj.shot, dict) else {}
            scene_id = shot.get("scene_id", "")
            chars = set(shot.get("characters_in_frame", []))

            if scene_id not in used_scenes:
                predicted.append(sid)
                used_scenes.add(scene_id)
                used_chars[scene_id] = chars
            elif not chars & used_chars.get(scene_id, set()):
                predicted.append(sid)
                used_chars[scene_id] |= chars
            elif safety >= 3:
                predicted.append(sid)
                used_chars[scene_id] |= chars

        return {
            "status": "ok",
            "predicted_batch": predicted,
            "predicted_size": len(predicted),
            "estimated_vram_gb": len(predicted) * self.GPU_VRAM_PER_SHOT,
            "ready_pool_size": len(ready),
            "dynamic_batch_limit": self._dynamic_batch_limit,
        }

    # ── Prometheus 指标导出 ─────────────────────────

    def _export_prometheus_metrics(self) -> dict:
        """导出 Prometheus 格式的管线指标, 供 Grafana 消费。

        指标列表:
          director_factory_pipeline_progress_pct
          director_factory_shots_total / _completed
          director_factory_concurrent_shots / _pending / _blocked
          director_factory_elapsed_seconds
          director_factory_eta_remaining_seconds
          director_factory_gpu_vram_used_gb
          director_factory_quality_score
          director_factory_retries_total
          director_factory_throughput_per_min
        """
        stats = self._compute_stats()
        eta = self._estimate_eta()

        return {
            "director_factory_pipeline_progress_pct": stats["progress_pct"],
            "director_factory_shots_total": stats["total_shots"],
            "director_factory_shots_completed": stats["completed"],
            "director_factory_concurrent_shots": sum(
                1 for s in self._shot_status if self._shot_status[s] == ShotStatus.IN_PROGRESS.value
            ),
            "director_factory_pending_shots": sum(
                1 for s in self._shot_status if self._shot_status[s] == ShotStatus.PENDING.value
            ),
            "director_factory_blocked_shots": stats["blocked_currently"],
            "director_factory_elapsed_seconds": round(time.time() - self._stats["start_time"], 1),
            "director_factory_eta_remaining_seconds": eta["estimated_remaining_seconds"],
            "director_factory_gpu_vram_used_gb": round(
                stats["avg_batch_size"] * self.GPU_VRAM_PER_SHOT, 1
            ),
            "director_factory_gpu_utilization_pct": stats["gpu_utilization"],
            "director_factory_vram_utilization_pct": round(
                stats["avg_batch_size"] / self.MAX_CONCURRENT_BY_VRAM * 100, 1
            ),
            "director_factory_throughput_per_min": stats["throughput_per_min"],
            "director_factory_retries_total": stats["retries_exhausted"],
            "director_factory_avg_shot_duration_s": round(self._avg_shot_duration, 1),
            "director_factory_active_projects": 1 if stats["completed"] < stats["total_shots"] else 0,
            "director_factory_critical_path_remaining": len(
                [s for s in self._critical_path if s not in self._completed]
            ),
        }
