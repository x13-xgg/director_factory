"""AssetDB 测试 — Memory/PostgreSQL 双后端 CRUD, 搜索, 生命周期"""

import asyncio
import json
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.tools.asset_db import AssetDB, asset_db
from src.core.config import config


# ── Fixtures ──────────────────────────────────────────

@pytest.fixture
def db():
    """每次测试创建独立的 AssetDB 实例"""
    db = AssetDB(base_dir="assets/test_db")
    yield db
    # cleanup
    import shutil
    index = Path("assets/test_db/_index.json")
    if index.exists():
        index.unlink()
    try:
        shutil.rmtree("assets/test_db")
    except Exception:
        pass


# ── Memory 后端 CRUD 测试 ──────────────────────────────

@pytest.mark.asyncio
async def test_put_and_get():
    """测试基本 put/get 操作"""
    db = AssetDB(base_dir="assets/test_put_get")
    record = db.put("script_repo", "test_001", {"title": "测试剧本", "scenes": 3})

    assert record["key"] == "test_001"
    assert record["data"]["title"] == "测试剧本"
    assert record["version"] == 1
    assert record["locked"] is False

    retrieved = db.get("script_repo", "test_001")
    assert retrieved is not None
    assert retrieved["data"]["title"] == "测试剧本"

    # cleanup
    import shutil
    try:
        shutil.rmtree("assets/test_put_get")
    except Exception:
        pass

    print("  ✓ test_put_and_get")


@pytest.mark.asyncio
async def test_put_version_increment():
    """测试重复 put 时版本号递增"""
    db = AssetDB(base_dir="assets/test_version")
    db.put("script_repo", "v1", {"data": "original"})
    record = db.put("script_repo", "v1", {"data": "updated"})

    assert record["version"] == 2

    import shutil
    try:
        shutil.rmtree("assets/test_version")
    except Exception:
        pass

    print("  ✓ test_version_increment")


@pytest.mark.asyncio
async def test_lock_and_is_locked():
    """测试资源锁定"""
    db = AssetDB(base_dir="assets/test_lock")
    db.put("char_asset_db", "hero", {"name": "Hero"})

    assert db.is_locked("char_asset_db", "hero") is False

    db.lock("char_asset_db", "hero")
    assert db.is_locked("char_asset_db", "hero") is True

    # 锁后再次 put 应保持 locked 状态
    record = db.put("char_asset_db", "hero", {"name": "Hero v2"})
    assert record["locked"] is True

    import shutil
    try:
        shutil.rmtree("assets/test_lock")
    except Exception:
        pass

    print("  ✓ test_lock_and_is_locked")


@pytest.mark.asyncio
async def test_version_tracking():
    """测试版本号追踪"""
    db = AssetDB(base_dir="assets/test_version2")
    db.put("prompt_repo", "p1", {"prompt": "v1"})
    assert db.version("prompt_repo", "p1") == 1

    db.put("prompt_repo", "p1", {"prompt": "v2"})
    assert db.version("prompt_repo", "p1") == 2

    assert db.version("prompt_repo", "nonexistent") == 0

    import shutil
    try:
        shutil.rmtree("assets/test_version2")
    except Exception:
        pass

    print("  ✓ test_version_tracking")


# ── 搜索测试 ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_memory():
    """测试内存后端搜索"""
    db = AssetDB(base_dir="assets/test_search")

    db.put("script_repo", "s1", {"title": "末日中的希望", "genre": "sci-fi"})
    db.put("script_repo", "s2", {"title": "雨夜的告别", "genre": "drama"})
    db.put("script_repo", "s3", {"title": "科幻冒险", "genre": "sci-fi"})

    results = db.search("script_repo", "末日")
    assert len(results) >= 1
    assert results[0]["key"] == "s1"

    results = db.search("script_repo", "sci-fi")
    assert len(results) == 2

    results = db.search("script_repo", "nonexistent_query_xyz")
    assert len(results) == 0

    import shutil
    try:
        shutil.rmtree("assets/test_search")
    except Exception:
        pass

    print("  ✓ test_search_memory")


@pytest.mark.asyncio
async def test_search_multi_keyword():
    """测试多关键词搜索"""
    db = AssetDB(base_dir="assets/test_search2")

    db.put("script_repo", "a", {"description": "一个机器人在废墟中寻找生命的痕迹"})
    db.put("script_repo", "b", {"description": "雨中的邂逅"})
    db.put("script_repo", "c", {"description": "机器人与人类的友谊"})

    results = db.search("script_repo", "机器人 废墟")
    assert len(results) == 2  # a 和 c
    keys = {r["key"] for r in results}
    assert "a" in keys
    assert "c" in keys

    import shutil
    try:
        shutil.rmtree("assets/test_search2")
    except Exception:
        pass

    print("  ✓ test_search_multi_keyword")


# ── PostgreSQL 后端连接测试 ───────────────────────────

@pytest.mark.asyncio
async def test_use_pg_false_when_memory_backend():
    """测试 memory 后端时 use_pg 返回 False"""
    db = AssetDB()
    assert db.use_pg is False
    assert db.backend == "memory"
    print("  ✓ test_use_pg_false_when_memory_backend")


@pytest.mark.asyncio
async def test_try_init_pg_noop_when_memory_backend():
    """测试 memory 后端时 _try_init_pg 不尝试连接"""
    db = AssetDB()
    result = await db._try_init_pg()
    assert result is False
    assert db._pg_available is False
    print("  ✓ test_try_init_pg_noop_when_memory_backend")


@pytest.mark.asyncio
async def test_pg_search_falls_back_to_memory():
    """测试 PG 不可用时 pg_search 回退到内存搜索"""
    db = AssetDB(base_dir="assets/test_pg_fallback")
    db.put("script_repo", "k1", {"title": "测试数据"})

    results = await db.pg_search("script_repo", "测试")
    assert len(results) >= 1
    assert results[0]["key"] == "k1"

    import shutil
    try:
        shutil.rmtree("assets/test_pg_fallback")
    except Exception:
        pass

    print("  ✓ test_pg_search_falls_back_to_memory")


@pytest.mark.asyncio
async def test_pg_load_all_noop_when_memory_backend():
    """测试 memory 后端时 pg_load_all 无操作且不抛异常"""
    db = AssetDB()
    await db.pg_load_all()  # 不应抛异常
    print("  ✓ test_pg_load_all_noop_when_memory_backend")


@pytest.mark.asyncio
async def test_close_noop_when_no_pool():
    """测试无连接池时 close 无操作且不抛异常"""
    db = AssetDB()
    await db.close()  # 不应抛异常
    assert AssetDB._PG_POOL is None
    print("  ✓ test_close_noop_when_no_pool")


# ── PostgreSQL 模拟测试 ────────────────────────────────

@pytest.mark.asyncio
async def test_pg_put_with_mock_pool():
    """测试 PG 写入路径 (mock asyncpg)"""
    mock_conn = MagicMock()
    mock_conn.execute = AsyncMock()
    mock_pool = MagicMock()
    mock_pool.acquire = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch.object(AssetDB, "_PG_POOL", mock_pool):
        db = AssetDB(base_dir="assets/test_mock_pg")
        db._pg_available = True

        record = {
            "key": "k1",
            "data": {"title": "test"},
            "metadata": {"author": "test"},
            "created_at": time.time(),
            "version": 1,
            "locked": False,
        }

        await db._pg_put("script_repo", "k1", record)
        # 验证 PG execute 被调用
        mock_conn.execute.assert_called_once()

    import shutil
    try:
        shutil.rmtree("assets/test_mock_pg")
    except Exception:
        pass

    print("  ✓ test_pg_put_with_mock_pool")


@pytest.mark.asyncio
async def test_pg_put_graceful_degradation():
    """测试 _pg_put 在 _try_init_pg 返回 False 时优雅降级"""
    db = AssetDB(base_dir="assets/test_pg_graceful")

    # _pg_available=False 时, _try_init_pg 返回 False, _pg_put 提前返回 None
    db._pg_available = False
    result = await db._pg_put("script_repo", "k1", {
        "key": "k1", "data": {}, "metadata": {}, "created_at": time.time(),
        "version": 1, "locked": False,
    })
    assert result is None  # _try_init_pg 返回 False, 提前返回

    import shutil
    try:
        shutil.rmtree("assets/test_pg_graceful")
    except Exception:
        pass

    print("  ✓ test_pg_put_graceful_degradation")


@pytest.mark.asyncio
async def test_pg_load_all_with_mock_data():
    """测试 pg_load_all 从 PG 加载数据到内存 (mock)"""
    mock_rows = [
        {
            "db_type": "script_repo",
            "key": "s1",
            "data": {"title": "PG 剧本"},
            "metadata": {},
            "created_at": time.time(),
            "version": 1,
            "locked": False,
        },
        {
            "db_type": "char_asset_db",
            "key": "hero",
            "data": {"name": "PG Hero"},
            "metadata": {},
            "created_at": time.time(),
            "version": 2,
            "locked": True,
        },
    ]

    mock_conn = MagicMock()
    mock_conn.fetch = AsyncMock(return_value=mock_rows)
    mock_pool = MagicMock()
    mock_pool.acquire = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch.object(AssetDB, "_PG_POOL", mock_pool):
        db = AssetDB(base_dir="assets/test_load")
        db._pg_available = True

        await db.pg_load_all()

        # 验证数据加载到内存
        assert db.get("script_repo", "s1") is not None
        assert db.get("script_repo", "s1")["data"]["title"] == "PG 剧本"
        assert db.get("char_asset_db", "hero")["version"] == 2
        assert db.is_locked("char_asset_db", "hero") is True

    import shutil
    try:
        shutil.rmtree("assets/test_load")
    except Exception:
        pass

    print("  ✓ test_pg_load_all_with_mock_data")


@pytest.mark.asyncio
async def test_ensure_pg_tables_creates_table():
    """测试 _ensure_pg_tables 执行 CREATE TABLE IF NOT EXISTS"""
    mock_conn = MagicMock()
    mock_conn.execute = AsyncMock()
    mock_pool = MagicMock()
    mock_pool.acquire = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch.object(AssetDB, "_PG_POOL", mock_pool):
        db = AssetDB()
        await db._ensure_pg_tables()

        # 验证 CREATE TABLE 被调用
        call_args = mock_conn.execute.call_args[0][0]
        assert "CREATE TABLE IF NOT EXISTS asset_records" in call_args
        assert "UNIQUE(db_type, key)" in call_args

    print("  ✓ test_ensure_pg_tables_creates_table")


# ── 全局单例测试 ──────────────────────────────────────

@pytest.mark.asyncio
async def test_global_asset_db_singleton():
    """测试全局 asset_db 单例可用"""
    from src.tools.asset_db import asset_db as global_db
    assert isinstance(global_db, AssetDB)
    assert global_db.backend == config.database.backend
    print("  ✓ test_global_asset_db_singleton")


# ── 四个分库测试 ──────────────────────────────────────

@pytest.mark.asyncio
async def test_four_partitions_independent():
    """测试四个逻辑分库相互独立"""
    db = AssetDB(base_dir="assets/test_partitions")

    db.put("script_repo", "shared_key", {"type": "script"})
    db.put("char_asset_db", "shared_key", {"type": "character"})
    db.put("prompt_repo", "shared_key", {"type": "prompt"})
    db.put("sfx_library", "shared_key", {"type": "sfx"})

    assert db.get("script_repo", "shared_key")["data"]["type"] == "script"
    assert db.get("char_asset_db", "shared_key")["data"]["type"] == "character"
    assert db.get("prompt_repo", "shared_key")["data"]["type"] == "prompt"
    assert db.get("sfx_library", "shared_key")["data"]["type"] == "sfx"

    # 删除一个不影响其他
    import shutil
    try:
        shutil.rmtree("assets/test_partitions")
    except Exception:
        pass

    print("  ✓ test_four_partitions_independent")


@pytest.mark.asyncio
async def test_persistence_index_file():
    """测试 _persist 生成 _index.json"""
    db = AssetDB(base_dir="assets/test_persist")
    db.put("script_repo", "s1", {"title": "Test"})

    index_path = Path("assets/test_persist/_index.json")
    assert index_path.exists()

    content = json.loads(index_path.read_text())
    assert "script_repo" in content
    assert content["script_repo"]["s1"]["version"] == 1

    import shutil
    try:
        shutil.rmtree("assets/test_persist")
    except Exception:
        pass

    print("  ✓ test_persistence_index_file")


# ── 压力/并发测试 ─────────────────────────────────────

@pytest.mark.asyncio
async def test_concurrent_puts():
    """测试并发 put 操作"""
    db = AssetDB(base_dir="assets/test_concurrent")

    async def put_item(i):
        db.put("script_repo", f"concurrent_{i}", {"index": i})

    tasks = [put_item(i) for i in range(20)]
    await asyncio.gather(*tasks)

    for i in range(20):
        assert db.get("script_repo", f"concurrent_{i}") is not None
        assert db.get("script_repo", f"concurrent_{i}")["data"]["index"] == i

    import shutil
    try:
        shutil.rmtree("assets/test_concurrent")
    except Exception:
        pass

    print("  ✓ test_concurrent_puts")


@pytest.mark.asyncio
async def test_large_record():
    """测试大记录存储"""
    db = AssetDB(base_dir="assets/test_large")
    large_data = {
        "title": "大型剧本",
        "scenes": [
            {
                "id": f"scene_{i}",
                "description": f"场景 {i} 的详细描述 " * 10,
                "characters": [f"角色_{j}" for j in range(10)],
                "shots": [{"shot_id": s, "action": f"动作_{s}" * 20} for s in range(10)],
            }
            for i in range(20)
        ],
    }

    record = db.put("script_repo", "large", large_data)
    assert record["version"] == 1

    retrieved = db.get("script_repo", "large")
    assert retrieved["data"]["title"] == "大型剧本"
    assert len(retrieved["data"]["scenes"]) == 20

    import shutil
    try:
        shutil.rmtree("assets/test_large")
    except Exception:
        pass

    print("  ✓ test_large_record")
