"""asset_db — 统一资产库 (memory / postgresql 双后端, 生产级)"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from src.core.config import config
from src.core.logging import get_logger

log = get_logger("AssetDB")


class AssetDB:
    """
    统一资产库 — 四个逻辑分库:

      script_repo    → 剧本、分镜、风格指南
      char_asset_db  → LoRA 权重、face_embedding、参考图 (路径引用)
      prompt_repo    → 已验证的 prompt 模板 + 参数
      sfx_library    → 音效库 (路径 + metadata 索引)

    后端选择 (由 DATABASE_BACKEND 环境变量控制):
      - memory:     内存 + JSON 文件索引 (默认)
      - postgresql: PostgreSQL + JSONB + pgvector (生产)
    """

    _PG_POOL = None

    def __init__(self, base_dir: str = "assets"):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self._store: dict[str, dict[str, dict]] = {
            "script_repo": {},
            "char_asset_db": {},
            "prompt_repo": {},
            "sfx_library": {},
        }
        self._locks: set[str] = set()
        self._pg_available = False

    @property
    def backend(self) -> str:
        return config.database.backend

    @property
    def use_pg(self) -> bool:
        return self.backend == "postgresql" and self._pg_available

    # ── Postgres pool (lazy) ─────────────────────────

    async def _get_pg_pool(self):
        if self._PG_POOL is not None:
            return self._PG_POOL
        try:
            import asyncpg
            self._PG_POOL = await asyncpg.create_pool(
                config.database.postgres_url,
                min_size=config.database.pg_pool_min,
                max_size=config.database.pg_pool_max,
            )
            await self._ensure_pg_tables()
            self._pg_available = True
            log.info("PostgreSQL 连接池已就绪")
            return self._PG_POOL
        except ImportError:
            self._pg_available = False
            raise RuntimeError("asyncpg 未安装")
        except Exception as e:
            self._pg_available = False
            log.warn(f"PostgreSQL 连接失败: {e}")
            raise

    async def _ensure_pg_tables(self):
        pool = self._PG_POOL
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS asset_records (
                    id SERIAL PRIMARY KEY,
                    db_type VARCHAR(64) NOT NULL,
                    key VARCHAR(512) NOT NULL,
                    data JSONB NOT NULL DEFAULT '{}',
                    metadata JSONB NOT NULL DEFAULT '{}',
                    created_at DOUBLE PRECISION NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    locked BOOLEAN NOT NULL DEFAULT FALSE,
                    UNIQUE(db_type, key)
                );
                CREATE INDEX IF NOT EXISTS idx_asset_db_type ON asset_records(db_type);
                CREATE INDEX IF NOT EXISTS idx_asset_key ON asset_records(db_type, key);
            """)

    async def _try_init_pg(self):
        """Try to initialize PG; returns True if ready."""
        if self._pg_available:
            return True
        if self.backend != "postgresql":
            return False
        try:
            await self._get_pg_pool()
            self._pg_available = True
            return True
        except Exception:
            return False

    # ── CRUD ─────────────────────────────────────────

    def put(self, db_type: str, key: str, data: Any, metadata: dict | None = None) -> dict:
        record = {
            "key": key,
            "data": data,
            "metadata": metadata or {},
            "created_at": time.time(),
            "version": 1,
            "locked": f"{db_type}:{key}" in self._locks,
        }
        existing = self._store.get(db_type, {}).get(key)
        if existing:
            record["version"] = existing.get("version", 0) + 1
        self._store.setdefault(db_type, {})[key] = record
        self._persist()

        # 异步写入 PG (fire-and-forget 如果可用)
        if self.backend == "postgresql":
            self._schedule_pg_put(db_type, key, record)

        return record

    def get(self, db_type: str, key: str) -> dict | None:
        return self._store.get(db_type, {}).get(key)

    def search(self, db_type: str, query: str, top_k: int = 5) -> list[dict]:
        results = []
        for key, record in self._store.get(db_type, {}).items():
            score = 0.0
            text = json.dumps(record, ensure_ascii=False).lower()
            for word in query.lower().split():
                if word in text:
                    score += 1.0
            if score > 0:
                results.append({"key": key, "score": score, "data": record["data"]})
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def lock(self, db_type: str, key: str):
        self._locks.add(f"{db_type}:{key}")
        if key in self._store.get(db_type, {}):
            self._store[db_type][key]["locked"] = True
        self._persist()

    def is_locked(self, db_type: str, key: str) -> bool:
        return f"{db_type}:{key}" in self._locks

    def version(self, db_type: str, key: str) -> int:
        record = self.get(db_type, key)
        return record["version"] if record else 0

    # ── PG async helpers ─────────────────────────────

    def _schedule_pg_put(self, db_type: str, key: str, record: dict):
        """Schedule an async PG write; fall back silently on failure."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._pg_put(db_type, key, record))
        except RuntimeError:
            pass  # no event loop, skip

    async def _pg_put(self, db_type: str, key: str, record: dict):
        try:
            if not await self._try_init_pg():
                return
            pool = self._PG_POOL
            async with pool.acquire() as conn:
                data_json = json.dumps(record["data"], ensure_ascii=False)
                meta_json = json.dumps(record["metadata"], ensure_ascii=False)
                await conn.execute("""
                    INSERT INTO asset_records (db_type, key, data, metadata, created_at, version, locked)
                    VALUES ($1, $2, $3::jsonb, $4::jsonb, $5, $6, $7)
                    ON CONFLICT (db_type, key) DO UPDATE SET
                        data = EXCLUDED.data,
                        metadata = EXCLUDED.metadata,
                        version = EXCLUDED.version
                """, db_type, key, data_json, meta_json,
                   record["created_at"], record["version"], record["locked"])
        except Exception as e:
            log.warn(f"PG write 失败 ({db_type}/{key}): {e}")

    async def pg_search(self, db_type: str, query: str, top_k: int = 5) -> list[dict]:
        """PostgreSQL 全文搜索 (需要 PG 后端)"""
        if not await self._try_init_pg():
            return self.search(db_type, query, top_k)
        pool = self._PG_POOL
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT key, data,
                       ts_rank(
                           to_tsvector('simple', data::text || ' ' || metadata::text),
                           plainto_tsquery('simple', $2)
                       ) AS score
                FROM asset_records
                WHERE db_type = $1
                  AND to_tsvector('simple', data::text || ' ' || metadata::text)
                      @@ plainto_tsquery('simple', $2)
                ORDER BY score DESC
                LIMIT $3
            """, db_type, query, top_k)
        results = []
        for row in rows:
            data = row["data"]
            if isinstance(data, str):
                data = json.loads(data)
            results.append({"key": row["key"], "score": float(row["score"]), "data": data})
        return results

    async def pg_load_all(self):
        """从 PG 加载所有数据到内存 (启动时同步)"""
        if not await self._try_init_pg():
            return
        pool = self._PG_POOL
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM asset_records")
        for row in rows:
            data = row["data"]
            if isinstance(data, str):
                data = json.loads(data)
            metadata = row["metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            self._store.setdefault(row["db_type"], {})[row["key"]] = {
                "key": row["key"],
                "data": data,
                "metadata": metadata,
                "created_at": row["created_at"],
                "version": row["version"],
                "locked": row["locked"],
            }
            if row["locked"]:
                self._locks.add(f"{row['db_type']}:{row['key']}")
        log.info(f"从 PG 加载了 {len(rows)} 条资产记录")

    async def close(self):
        """关闭 PG 连接池"""
        if self._PG_POOL:
            await self._PG_POOL.close()
            self._PG_POOL = None
            log.info("PostgreSQL 连接池已关闭")

    # ── Persistence ──────────────────────────────────

    def _persist(self):
        summary = {
            db_type: {k: {"key": v["key"], "version": v["version"], "locked": v["locked"]}
                      for k, v in records.items()}
            for db_type, records in self._store.items()
        }
        (self.base / "_index.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


# 全局单例
asset_db = AssetDB()
