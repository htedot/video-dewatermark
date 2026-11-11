"""BaseAgent：所有业务 Agent 的契约基类。

硬契约（规划文档 3.2）：
* 每个 Agent 只实现 ``run(payload) -> payload``；
* 上游产物 **只从 payload 取**，本环节产物 **只往 payload 放**；
* **禁止 Agent 之间互相 import** —— 想通信就发事件（``self.bus``）或改 payload；
* 一个文件一个 Agent。

关于 async：LLM 调用必须是 ``await``（见 3.3 的 ``text, _ = await ...``），
所以 ``run`` 写成协程；CPU 密集的 OpenCV/ffmpeg 一律 ``await asyncio.to_thread(...)``
丢到线程池，事件循环不会卡住 SSE。``Orchestrator`` 对同步/异步 ``run`` 都能调度。
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import config
import llm as llm_module
from event_bus import EventBus, bus as global_bus


class AgentError(RuntimeError):
    """Agent 执行失败（不可降级的错误）。"""


class BaseAgent:
    #: 节点 key，与 orchestrator 注册名一致
    name: str = "base"
    #: 中文短名（前端步骤条显示）
    label: str = "基础 Agent"
    #: 一句话职责（README/前端提示）
    description: str = ""

    def __init__(
        self,
        bus: EventBus | None = None,
        llm: llm_module.LLMClient | None = None,
    ) -> None:
        self.bus: EventBus = bus or global_bus
        self.llm: llm_module.LLMClient = llm or llm_module.get_client()

    # ------------------------------------------------------------------ #
    # 子类只需要实现这个
    # ------------------------------------------------------------------ #
    async def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # 事件/进度辅助
    # ------------------------------------------------------------------ #
    @staticmethod
    def run_id(payload: dict[str, Any]) -> str:
        return str(payload.get("run_id", "unknown"))

    @staticmethod
    def run_dir(payload: dict[str, Any]) -> Path:
        return Path(payload["run_dir"])

    def emit(self, payload: dict[str, Any], event_type: str, message: str = "", **data: Any) -> None:
        self.bus.emit(event_type, self.run_id(payload), message, agent=self.name, **data)

    def started(self, payload: dict[str, Any], message: str | None = None) -> None:
        self.emit(
            payload,
            "agent_started",
            message or f"{self.label} 开始",
            label=self.label,
            attempt=payload.get("attempt", 0),
        )

    def progress(
        self,
        payload: dict[str, Any],
        message: str,
        percent: float | None = None,
        **data: Any,
    ) -> None:
        extra = dict(data)
        if percent is not None:
            extra["percent"] = round(float(percent), 1)
        self.emit(payload, "agent_progress", message, label=self.label, **extra)

    def completed(self, payload: dict[str, Any], message: str = "", **data: Any) -> None:
        self.emit(
            payload,
            "agent_completed",
            message or f"{self.label} 完成",
            label=self.label,
            attempt=payload.get("attempt", 0),
            **data,
        )

    def degraded(self, payload: dict[str, Any], reason: str, **data: Any) -> None:
        """LLM 不可用/失败时的规则化降级：记一笔，流水线继续。"""
        self.llm.note_degraded(reason)
        self.emit(payload, "llm_call", f"LLM 降级为规则：{reason}", ok=False, **data)

    async def call_llm(
        self,
        payload: dict[str, Any],
        purpose: str,
        system: str,
        user: str,
        parse: Callable[[Any], Any] | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> tuple[Any, int]:
        """统一的 LLM 调用入口：发事件 + 记录 token + 失败转 ``None``。

        返回值 ``(result, tokens)``；``result is None`` 表示这次降级了，
        调用方必须自己走规则路径（**任何 LLM 环节都不允许因失败而崩流水线**）。
        """
        if not self.llm.available:
            self.degraded(payload, f"未配置 DEEPSEEK_API_KEY（{purpose} 用规则）", purpose=purpose)
            return None, 0
        started = time.perf_counter()
        try:
            if json_mode:
                data, tokens = await self.llm.complete_json(system, user, max_tokens=max_tokens)
            else:
                data, tokens = await self.llm.complete_text(system, user, max_tokens=max_tokens)
            parsed = parse(data) if parse else data
            self.emit(
                payload,
                "llm_call",
                f"LLM 调用成功（{purpose}，{tokens} tokens）",
                ok=True,
                purpose=purpose,
                tokens=tokens,
                latency=round(time.perf_counter() - started, 2),
            )
            return parsed, tokens
        except llm_module.LLMUnavailable as exc:
            self.degraded(payload, f"{exc}（{purpose} 用规则）", purpose=purpose)
            return None, 0
        except llm_module.LLMError as exc:
            self.degraded(payload, f"{purpose} 调用失败：{exc}", purpose=purpose)
            return None, 0

    # ------------------------------------------------------------------ #
    # 小工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def load_json(name: str) -> dict[str, Any]:
        """读 ``backend/data/<name>`` 规则库。缺文件返回空 dict（规则库可选）。"""
        path = config.DATA_DIR / name
        if not path.is_file():
            return {}
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def read_prompt(name: str) -> str:
        """读 ``backend/data/prompts/<name>``（系统提示词单块存文件）。"""
        path = config.PROMPT_DIR / name
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8").strip()

    @staticmethod
    def write_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

    @staticmethod
    async def to_thread(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """把 CPU 密集活丢线程池，保持事件循环呼吸（SSE 才能持续推）。"""
        return await asyncio.to_thread(func, *args, **kwargs)

    async def sleep(self, seconds: float) -> Awaitable[None]:  # pragma: no cover - 便捷
        await asyncio.sleep(seconds)


class PayloadDict(dict):
    """带链路痕迹的 payload（便于报告里还原每一步耗时）。"""

    def stamp(self, agent: str, status: str, seconds: float, **extra: Any) -> None:
        trace = self.setdefault("trace", [])
        item = {"agent": agent, "status": status, "seconds": round(seconds, 3)}
        item.update(extra)
        trace.append(item)

    def stamp_start(self, agent: str) -> float:
        return time.perf_counter()

    def stamp_end(self, agent: str, started: float, **extra: Any) -> None:
        self.stamp(agent, "ok", time.perf_counter() - started, **extra)