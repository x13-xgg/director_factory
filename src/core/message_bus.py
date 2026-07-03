"""消息总线 — Agent 间通信 (memory / redis / nats 三后端, 生产级)"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from src.core.config import config
from src.core.logging import get_logger

log = get_logger("MessageBus")


@dataclass
class Message:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    sender: str = ""
    target: str = ""          # "" = broadcast
    msg_type: str = ""
    payload: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    in_reply_to: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id, "sender": self.sender, "target": self.target,
            "msg_type": self.msg_type, "payload": self.payload,
            "timestamp": self.timestamp, "in_reply_to": self.in_reply_to,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        return cls(
            id=d.get("id", ""), sender=d.get("sender", ""), target=d.get("target", ""),
            msg_type=d.get("msg_type", ""), payload=d.get("payload", {}),
            timestamp=d.get("timestamp", 0), in_reply_to=d.get("in_reply_to", ""),
        )


@dataclass
class Envelope:
    future: asyncio.Future
    timeout: float = 60.0


# ── Abstract backend ────────────────────────────────────


class _Backend:
    """消息总线后端抽象"""

    async def register_agent(self, agent_id: str): ...
    async def send(self, msg: Message) -> None: ...
    async def receive(self, agent_id: str, timeout: float) -> Message | None: ...
    async def broadcast(self, msg: Message, exclude: list[str]) -> None: ...
    async def store_request(self, msg: Message, timeout: float) -> None: ...
    async def resolve_request(self, msg_id: str, result: dict) -> bool: ...
    async def publish(self, topic: str, msg: Message) -> None: ...
    async def subscribe(self, topic: str, handler: Callable[[Message], Coroutine]) -> None: ...
    async def close(self) -> None: ...


# ── Memory backend ───────────────────────────────────────


class _MemoryBackend(_Backend):
    def __init__(self):
        self._inboxes: dict[str, asyncio.Queue[Message]] = defaultdict(asyncio.Queue)
        self._subscriptions: dict[str, list[Callable]] = defaultdict(list)
        self._pending: dict[str, Envelope] = {}
        self._event_log: list[Message] = []

    async def register_agent(self, agent_id: str):
        self._inboxes[agent_id] = asyncio.Queue()

    async def send(self, msg: Message):
        await self._inboxes[msg.target].put(msg)
        self._event_log.append(msg)

    async def receive(self, agent_id: str, timeout: float) -> Message | None:
        try:
            return await asyncio.wait_for(self._inboxes[agent_id].get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def broadcast(self, msg: Message, exclude: list[str]):
        for aid, inbox in self._inboxes.items():
            if aid not in exclude and aid != msg.sender:
                await inbox.put(msg)
        self._event_log.append(msg)

    async def store_request(self, msg: Message, timeout: float):
        loop = asyncio.get_event_loop()
        self._pending[msg.id] = Envelope(future=loop.create_future(), timeout=timeout)

    async def resolve_request(self, msg_id: str, result: dict) -> bool:
        env = self._pending.pop(msg_id, None)
        if env and not env.future.done():
            env.future.set_result(result)
            return True
        return False

    async def publish(self, topic: str, msg: Message):
        self._event_log.append(msg)
        for handler in self._subscriptions.get(topic, []):
            try:
                await handler(msg)
            except Exception:
                pass

    async def subscribe(self, topic: str, handler: Callable[[Message], Coroutine]):
        self._subscriptions[topic].append(handler)

    async def close(self):
        pass

    def get_events(self, topic: str | None, limit: int) -> list[Message]:
        if topic:
            return [m for m in self._event_log if m.target == topic][-limit:]
        return self._event_log[-limit:]

    def clear(self):
        self._inboxes.clear()
        self._subscriptions.clear()
        self._pending.clear()
        self._event_log.clear()


# ── Redis backend ────────────────────────────────────────


class _RedisBackend(_Backend):
    """Redis Streams 后端 — 持久化消息队列"""

    def __init__(self, redis_url: str = ""):
        self._redis_url = redis_url or config.message_bus.redis_url
        self._client = None
        self._maxlen = config.message_bus.redis_stream_maxlen
        self._subscriptions: dict[str, list[Callable]] = defaultdict(list)
        self._listener_task: asyncio.Task | None = None

    async def _get_client(self):
        if self._client is None:
            try:
                import redis.asyncio as aioredis
                # 尝试连接真实 Redis
                client = aioredis.from_url(self._redis_url, decode_responses=False)
                await client.ping()
                self._client = client
                log.info(f"Redis 已连接: {self._redis_url}")
            except ImportError:
                raise RuntimeError("redis 库未安装。pip install redis")
            except Exception as e:
                # 回退到 fakeredis (内存模拟, 零依赖)
                try:
                    import fakeredis.aioredis as faioredis
                    self._client = faioredis.FakeRedis(decode_responses=False)
                    await self._client.ping()
                    log.info(f"Redis 不可用 ({e})，使用 fakeredis 内存模拟")
                except ImportError:
                    self._client = None
                    raise RuntimeError(f"Redis 连接失败且 fakeredis 未安装: {e}")
                except Exception as fe:
                    self._client = None
                    raise RuntimeError(f"Redis 连接失败 ({e}), fakeredis 也失败: {fe}")
        return self._client

    def _stream_key(self, agent_id: str) -> str:
        return f"df:inbox:{agent_id}"

    def _broadcast_channel(self) -> str:
        return "df:broadcast"

    async def register_agent(self, agent_id: str):
        r = await self._get_client()
        # 创建消费者组 (幂等)
        try:
            await r.xgroup_create(self._stream_key(agent_id), "df_group", id="0", mkstream=True)
        except Exception:
            pass

    async def send(self, msg: Message):
        r = await self._get_client()
        data = {"msg": json.dumps(msg.to_dict(), ensure_ascii=False)}
        await r.xadd(self._stream_key(msg.target), data, maxlen=self._maxlen)

    async def receive(self, agent_id: str, timeout: float) -> Message | None:
        r = await self._get_client()
        key = self._stream_key(agent_id)
        timeout_ms = max(int(timeout * 1000), 100)
        try:
            results = await r.xreadgroup("df_group", agent_id, {key: ">"}, count=1, block=timeout_ms)
        except Exception:
            # 消费者组可能不存在
            try:
                await r.xgroup_create(key, "df_group", id="0", mkstream=True)
            except Exception:
                pass
            results = await r.xreadgroup("df_group", agent_id, {key: ">"}, count=1, block=timeout_ms)

        if not results:
            return None
        for stream_key, entries in results:
            for msg_id, fields in entries:
                raw = fields.get(b"msg", fields.get("msg", "{}"))
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                data = json.loads(raw)
                return Message.from_dict(data)
        return None

    async def broadcast(self, msg: Message, exclude: list[str]):
        r = await self._get_client()
        data = json.dumps(msg.to_dict(), ensure_ascii=False)
        # 使用 pub/sub 广播
        await r.publish(self._broadcast_channel(), data)
        # 同时写入每个 agent 的 stream
        for agent_id in await r.keys("df:inbox:*"):
            aid = agent_id.decode("utf-8").split(":")[-1] if isinstance(agent_id, bytes) else agent_id.split(":")[-1]
            if aid not in exclude and aid != msg.sender:
                await r.xadd(self._stream_key(aid), {"msg": data}, maxlen=self._maxlen)

    async def store_request(self, msg: Message, timeout: float):
        r = await self._get_client()
        data = json.dumps({"timeout": timeout, "ts": time.time()}, ensure_ascii=False)
        await r.setex(f"df:request:{msg.id}", int(timeout), data)

    async def resolve_request(self, msg_id: str, result: dict) -> bool:
        r = await self._get_client()
        key = f"df:response:{msg_id}"
        await r.setex(key, 300, json.dumps(result, ensure_ascii=False))
        return True

    async def publish(self, topic: str, msg: Message):
        r = await self._get_client()
        data = json.dumps(msg.to_dict(), ensure_ascii=False)
        await r.publish(f"df:topic:{topic}", data)

    async def subscribe(self, topic: str, handler: Callable[[Message], Coroutine]):
        self._subscriptions[topic].append(handler)
        if self._listener_task is None:
            self._listener_task = asyncio.create_task(self._listen_pubsub())

    async def _listen_pubsub(self):
        try:
            r = await self._get_client()
            pubsub = r.pubsub()
            # 实际生产环境中应动态订阅 topic
            await pubsub.psubscribe("df:topic:*")
            async for message in pubsub.listen():
                if message["type"] == "pmessage":
                    topic = message["channel"].decode("utf-8").replace("df:topic:", "")
                    data = json.loads(message["data"].decode("utf-8"))
                    msg = Message.from_dict(data)
                    for handler in self._subscriptions.get(topic, []):
                        try:
                            await handler(msg)
                        except Exception:
                            pass
        except Exception as e:
            log.warn(f"PubSub 监听异常: {e}")

    async def close(self):
        if self._client:
            await self._client.close()
            self._client = None


# ── NATS backend ─────────────────────────────────────────


class _NatsBackend(_Backend):
    """NATS JetStream 后端 — 高性能消息系统"""

    def __init__(self, nats_url: str = ""):
        self._nats_url = nats_url or config.message_bus.nats_url
        self._prefix = config.message_bus.nats_subject_prefix
        self._nc = None
        self._js = None
        self._subscriptions: dict[str, list[Callable]] = defaultdict(list)

    async def _get_connection(self):
        if self._nc is None:
            try:
                import nats
                from nats.js import api as js_api
                self._nc = await nats.connect(self._nats_url)
                self._js = self._nc.jetstream()
                log.info(f"NATS 已连接: {self._nats_url}")
            except ImportError:
                raise RuntimeError("nats-py 库未安装。pip install nats-py")
            except Exception as e:
                self._nc = None
                raise RuntimeError(f"NATS 连接失败: {e}")
        return self._nc, self._js

    def _subject(self, *parts: str) -> str:
        return f"{self._prefix}.{'.'.join(parts)}"

    async def register_agent(self, agent_id: str):
        nc, js = await self._get_connection()
        # 为 agent 创建持久化 stream
        subject = self._subject("inbox", agent_id)
        try:
            await js.add_stream(name=f"df_inbox_{agent_id}", subjects=[subject])
        except Exception:
            pass

    async def send(self, msg: Message):
        nc, js = await self._get_connection()
        subject = self._subject("inbox", msg.target)
        data = json.dumps(msg.to_dict(), ensure_ascii=False).encode("utf-8")
        await nc.publish(subject, data)

    async def receive(self, agent_id: str, timeout: float) -> Message | None:
        nc, js = await self._get_connection()
        subject = self._subject("inbox", agent_id)
        try:
            sub = await js.pull_subscribe(subject, f"df_consumer_{agent_id}")
            msgs = await sub.fetch(1, timeout=timeout)
            for m in msgs:
                data = json.loads(m.data.decode("utf-8"))
                await m.ack()
                return Message.from_dict(data)
        except Exception:
            pass
        return None

    async def broadcast(self, msg: Message, exclude: list[str]):
        nc, js = await self._get_connection()
        subject = self._subject("broadcast")
        data = json.dumps(msg.to_dict(), ensure_ascii=False).encode("utf-8")
        await nc.publish(subject, data)

    async def store_request(self, msg: Message, timeout: float):
        pass  # NATS 原生支持 request-reply

    async def resolve_request(self, msg_id: str, result: dict) -> bool:
        return True

    async def publish(self, topic: str, msg: Message):
        nc, js = await self._get_connection()
        subject = self._subject("topic", topic)
        data = json.dumps(msg.to_dict(), ensure_ascii=False).encode("utf-8")
        await nc.publish(subject, data)

    async def subscribe(self, topic: str, handler: Callable[[Message], Coroutine]):
        self._subscriptions[topic].append(handler)
        nc, js = await self._get_connection()
        subject = self._subject("topic", topic)

        async def cb(msg):
            data = json.loads(msg.data.decode("utf-8"))
            message = Message.from_dict(data)
            for h in self._subscriptions.get(topic, []):
                try:
                    await h(message)
                except Exception:
                    pass

        await nc.subscribe(subject, cb=cb)

    async def close(self):
        if self._nc:
            await self._nc.close()
            self._nc = None


# ── MessageBus (facade) ──────────────────────────────────


class MessageBus:
    """
    消息总线 — Agent 间通信的中央调度

    后端选择 (由 MSG_BACKEND 环境变量控制):
      - memory: 内存队列 (开发/测试)
      - redis:  Redis Streams + Pub/Sub (中等规模生产)
      - nats:   NATS JetStream (高吞吐生产)

    通信模式: send | broadcast | request | publish
    """

    def __init__(self):
        self._backend = self._create_backend()
        self._agent_ids: set[str] = set()

    def _create_backend(self) -> _Backend:
        backend_type = config.message_bus.backend
        if backend_type == "redis":
            try:
                b = _RedisBackend()
                log.info("使用 Redis 消息总线后端")
                return b
            except Exception as e:
                log.warn(f"Redis 不可用 ({e})，回退到内存后端")
        elif backend_type == "nats":
            try:
                b = _NatsBackend()
                log.info("使用 NATS 消息总线后端")
                return b
            except Exception as e:
                log.warn(f"NATS 不可用 ({e})，回退到内存后端")
        log.info("使用内存消息总线后端")
        return _MemoryBackend()

    # ── Public API ──────────────────────────────────

    async def register_agent(self, agent_id: str):
        self._agent_ids.add(agent_id)
        await self._backend.register_agent(agent_id)

    async def send(self, sender: str, target: str, msg_type: str, payload: dict) -> Message:
        msg = Message(sender=sender, target=target, msg_type=msg_type, payload=payload)
        await self._backend.send(msg)
        return msg

    async def receive(self, agent_id: str, timeout: float = 5.0) -> Message | None:
        return await self._backend.receive(agent_id, timeout)

    async def broadcast(self, sender: str, msg_type: str, payload: dict, exclude: list[str] | None = None):
        msg = Message(sender=sender, target="*", msg_type=msg_type, payload=payload)
        await self._backend.broadcast(msg, exclude or [])

    async def request(self, sender: str, target: str, msg_type: str, payload: dict, timeout: float = 60.0) -> dict | None:
        msg = Message(sender=sender, target=target, msg_type=msg_type, payload=payload)
        await self._backend.store_request(msg, timeout)
        await self._backend.send(msg)

        # 轮询响应 (Redis/NATS 后端用)
        if isinstance(self._backend, _MemoryBackend):
            return await self._wait_memory_response(msg.id, timeout)
        else:
            return await self._poll_response(msg.id, timeout)

    async def reply(self, original_msg: Message, payload: dict):
        await self._backend.resolve_request(original_msg.id, payload)
        # 对于内存后端，直接设置 future
        if isinstance(self._backend, _MemoryBackend):
            env = self._backend._pending.pop(original_msg.id, None)
            if env and not env.future.done():
                env.future.set_result(payload)

    async def publish(self, topic: str, event: dict, sender: str = "system"):
        msg = Message(sender=sender, target=topic, msg_type="event", payload=event)
        await self._backend.publish(topic, msg)

    def subscribe(self, topic: str, handler: Callable[[Message], Coroutine]):
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._backend.subscribe(topic, handler))
        except RuntimeError:
            asyncio.run(self._backend.subscribe(topic, handler))

    # ── Internal ────────────────────────────────────

    async def _wait_memory_response(self, msg_id: str, timeout: float) -> dict | None:
        if not isinstance(self._backend, _MemoryBackend):
            return None
        env = self._backend._pending.get(msg_id)
        if env is None:
            return None
        try:
            return await asyncio.wait_for(env.future, timeout=timeout)
        except asyncio.TimeoutError:
            self._backend._pending.pop(msg_id, None)
            return None

    async def _poll_response(self, msg_id: str, timeout: float) -> dict | None:
        if isinstance(self._backend, _RedisBackend):
            r = await self._backend._get_client()
            key = f"df:response:{msg_id}"
            deadline = time.time() + timeout
            while time.time() < deadline:
                data = await r.get(key)
                if data:
                    await r.delete(key)
                    if isinstance(data, bytes):
                        data = data.decode("utf-8")
                    return json.loads(data)
                await asyncio.sleep(0.05)
        return None

    # ── Utility ─────────────────────────────────────

    def get_events(self, topic: str | None = None, limit: int = 100) -> list[Message]:
        if isinstance(self._backend, _MemoryBackend):
            if topic:
                return [m for m in self._backend._event_log if m.target == topic][-limit:]
            return self._backend._event_log[-limit:]
        return []

    def clear(self):
        if isinstance(self._backend, _MemoryBackend):
            self._backend._inboxes.clear()
            self._backend._subscriptions.clear()
            self._backend._pending.clear()
            self._backend._event_log.clear()

    async def close(self):
        await self._backend.close()
