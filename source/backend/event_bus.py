"""EventBus：Agent 之间、Agent 与前端之间唯一的通信通道。

设计要点（对应规划文档 3.2 四件套契约）：
* ``bus.publish(event)`` 内部完成 ``json.dumps``，返回值就是可以直接塞进 SSE 的字符串；
* ``bus.subscribe()`` 是**同步**方法，不能 await —— SSE 生成器里直接 ``q = bus.subscribe()``；
* 订阅者拿到的是 ``asyncio.Queue``，队列里放的是**已经序列化好的 raw 字符串**；
* 处理线程（OpenCV/ffmpeg 跑在 ``asyncio.to_thread`` 里）也会 publish，
  所以派发统一走 ``loop.call_soon_threadsafe``，跨线程安全；
* 每个事件带 ``run_id``，SSE 侧按 run_id 过滤，多任务并发不会串流。

本模块不 import 任何 Agent，Agent 也不 import 事件源，双向只认 dict 事件。
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any, Iterable


# 事件类型（前端按 type 决定怎么渲染）
EVENT_TYPES = (
    "task_queued",        # 任务进入队列
    "task_started",       # 流水线开始
    "agent_started",      # 某个 Agent 开始
    "agent_progress",     # Agent 内部进度（帧数/百分比/消息）
    "agent_completed",    # 某个 Agent 完成
    "agent_failed",       # 某个 Agent 抛错（有降级路径时不会终止任务）
    "llm_call",           # 一次 LLM 调用（成功/降级都发）
    "parallel_started",   # 并行聚合开始
    "parallel_completed", # 并行聚合结束
    "review_feedback",    # 质检不达标 → 回灌重生成
    "task_completed",     # 任务成功（带产物清单）
    "task_failed",        # 任务失败（带错误信息）
    "run_log",            # 兜底：一般性日志
)


class EventBus:
    """极简发布/订阅总线，只做「扇出 + 序列化」，不做落库。"""

    def __init__(self, keep_history: int = 400) -> None:
        # (queue, loop) —— loop 用于跨线程安全投递
        self._subs: list[tuple[asyncio.Queue[str], asyncio.AbstractEventLoop]] = []
        self._lock = threading.Lock()
        self._seq = 0
        self._history: list[str] = []
        self._keep_history = max(0, keep_history)

    # ------------------------------------------------------------------ #
    # 订阅（同步方法，不能 await）
    # ------------------------------------------------------------------ #
    def subscribe(self, maxsize: int = 2000) -> asyncio.Queue[str]:
        """注册一个订阅者。必须在事件循环线程里调用（SSE handler 里）。"""
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=maxsize)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # 理论上不会发生；退化成"只写入不派发"
            loop = None  # type: ignore[assignment]
        with self._lock:
            self._subs.append((queue, loop))  # type: ignore[arg-type]
        return queue

    def unsubscribe(self, queue: asyncio.Queue[str]) -> None:
        with self._lock:
            self._subs = [(q, lp) for (q, lp) in self._subs if q is not queue]

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subs)

    # ------------------------------------------------------------------ #
    # 发布
    # ------------------------------------------------------------------ #
    def publish(self, event: dict[str, Any]) -> str:
        """序列化并扇出。返回可以直接 ``yield f"data: {raw}\\n\\n"`` 的字符串。"""
        payload = dict(event)
        with self._lock:
            self._seq += 1
            payload.setdefault("seq", self._seq)
        payload.setdefault("ts", round(time.time(), 3))
        raw = json.dumps(payload, ensure_ascii=False, default=str)
        self._remember(raw)
        self._fanout(raw)
        return raw

    def emit(
        self,
        event_type: str,
        run_id: str | None = None,
        message: str = "",
        **data: Any,
    ) -> str:
        """publish 的语法糖：``bus.emit("agent_started", run_id, "开始检测", agent="detector")``。"""
        event: dict[str, Any] = {"type": event_type}
        if run_id is not None:
            event["run_id"] = run_id
        if message:
            event["message"] = message
        if data:
            event["data"] = data
        return self.publish(event)

    def _remember(self, raw: str) -> None:
        if self._keep_history <= 0:
            return
        with self._lock:
            self._history.append(raw)
            if len(self._history) > self._keep_history:
                del self._history[: len(self._history) - self._keep_history]

    def history(self, run_id: str | None = None, limit: int = 100) -> list[str]:
        """最近事件（调试/断线重连用）。"""
        with self._lock:
            items = list(self._history)
        if run_id:
            items = [raw for raw in items if f'"run_id": "{run_id}"' in raw]
        return items[-limit:]

    def _fanout(self, raw: str) -> None:
        with self._lock:
            targets = list(self._subs)
        for queue, loop in targets:
            try:
                if loop is not None and loop.is_running():
                    # 跨线程 / 同线程都安全：回调在 loop 里执行 put_nowait
                    loop.call_soon_threadsafe(self._safe_put, queue, raw)
                else:
                    self._safe_put(queue, raw)
            except RuntimeError:
                # loop 已关闭：这个订阅者已经走下坡路了，交给 finally 清理
                continue

    @staticmethod
    def _safe_put(queue: asyncio.Queue[str], raw: str) -> None:
        try:
            queue.put_nowait(raw)
        except asyncio.QueueFull:
            # 慢客户端不拖死流水线：丢弃一条最旧的，再塞新的
            try:
                queue.get_nowait()
                queue.put_nowait(raw)
            except Exception:  # pragma: no cover - 极端竞态
                pass


def format_sse(raw: str) -> str:
    """把总线 raw 字符串包成 SSE 帧。"""
    return f"data: {raw}\n\n"


def iter_sse(events: Iterable[str]) -> str:
    """把若干 raw 事件拼成一段 SSE 文本（测试用）。"""
    return "".join(format_sse(raw) for raw in events)


# 进程内单例：main.py 和 orchestrator 共享同一条总线
bus = EventBus()