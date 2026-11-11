"""LLM 客户端：DeepSeek 文本接口（任何 OpenAI 协议端点都可用）。

契约（规划文档 3.3 / 硬规矩 1）：
* ``complete_text()`` 返回 ``(text, token)`` 元组，调用处写 ``text, _ = await ...``；
* ``complete_json()`` 顶层只认 dict，要 list 用 ``complete_json_list()``；
* LLM 返回的 JSON 一律不可信 —— 必须 ``str()`` 之后用平衡括号扫描 ``_extract_json()``
  抠出最外层 JSON，**禁止** ``json.loads(llm_output)``；
* 所有调用带超时（默认 90s）+ 重试，失败抛 ``LLMError``，由调用方走规则化降级。

本模块只负责"把一段文字要回来"。判断/创作规则在 ``backend/data`` 的 JSON 里，
调用方拿到结果后必须自己做字段校验与原文回溯，不信 LLM 的自述。
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import config


class LLMError(RuntimeError):
    """所有 LLM 异常的统一类型（未配置 Key / 超时 / 网络错误 / 返回不可解析）。"""


class LLMUnavailable(LLMError):
    """没有可用 Key（或显式关闭）——调用方应立即走规则降级，不要重试。"""


class LLMUsage:
    """轻量统计，最终写进 detect_report.json / review_report.json。"""

    def __init__(self) -> None:
        self.calls = 0
        self.ok = 0
        self.failed = 0
        self.degraded = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.last_error: str | None = None
        self.last_latency: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "ok": self.ok,
            "failed": self.failed,
            "degraded": self.degraded,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "last_error": self.last_error,
            "last_latency": None if self.last_latency is None else round(self.last_latency, 2),
        }


# --------------------------------------------------------------------------- #
# JSON 提取：平衡括号扫描（不可信文本 → dict/list）
# --------------------------------------------------------------------------- #
def _extract_json(text: Any) -> dict | list | None:
    """从任意文本里抠出第一个**完整**的 JSON 对象/数组。

    * 逐个扫描 ``{`` / ``[`` 起点，按括号配对找到闭合位置（跳过字符串内的括号与转义）；
    * 对每个候选片段尝试 ``json.loads``，成功即返回；
    * 全部失败返回 ``None``，绝不抛异常给业务层。

    这是硬规矩 1 的落地：LLM 输出可能带 ```json 围栏、解释性前言、尾随逗号。
    """
    if text is None:
        return None
    if isinstance(text, (dict, list)):
        return text  # 已经是结构化对象（测试里的假客户端会这样返回）
    raw = str(text)
    if not raw.strip():
        return None

    open_to_close = {"{": "}", "[": "]"}
    length = len(raw)
    for start, opener in ((i, ch) for i, ch in enumerate(raw) if ch in open_to_close):
        closer = open_to_close[opener]
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, length):
            ch = raw[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    fragment = raw[start : idx + 1]
                    try:
                        return json.loads(fragment)
                    except (json.JSONDecodeError, ValueError):
                        break  # 这个起点废了，换下一个起点
        # 未闭合或解析失败 → 继续找下一个起点
    return None


def _extract_json_strict(text: Any, expect: type) -> Any:
    data = _extract_json(text)
    if isinstance(data, expect):
        return data
    raise LLMError(f"LLM 返回无法解析为 {expect.__name__}：{str(text)[:200]!r}")


# --------------------------------------------------------------------------- #
# 客户端
# --------------------------------------------------------------------------- #
class LLMClient:
    """OpenAI 协议客户端封装。没有 Key 时 ``available`` 为 False。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.api_key = (api_key if api_key is not None else config.DEEPSEEK_API_KEY).strip()
        self.base_url = (base_url or config.DEEPSEEK_BASE_URL).strip()
        self.model = (model or config.DEEPSEEK_MODEL).strip()
        self.timeout = float(timeout if timeout is not None else config.LLM_TIMEOUT)
        self.usage = LLMUsage()
        self._client: Any | None = None
        self._client_error: str | None = None

    # ------------------------------------------------------------------ #
    @property
    def available(self) -> bool:
        return bool(self.api_key)

    @property
    def status(self) -> dict[str, Any]:
        return {
            "configured": self.available,
            "base_url": self.base_url,
            "model": self.model,
            "timeout": self.timeout,
            "usage": self.usage.as_dict(),
            "client_error": self._client_error,
        }

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.available:
            raise LLMUnavailable("未配置 DEEPSEEK_API_KEY（backend/.env）")
        if self._client_error:
            raise LLMUnavailable(f"客户端初始化失败: {self._client_error}")
        try:
            from openai import AsyncOpenAI  # 延迟导入，没装也不影响规则降级

            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=0,  # 重试由本类控制，便于统计
            )
        except Exception as exc:  # pragma: no cover - 环境问题
            self._client_error = str(exc)
            raise LLMUnavailable(f"openai 客户端不可用: {exc}") from exc
        return self._client

    # ------------------------------------------------------------------ #
    async def _chat(
        self,
        system: str,
        user: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> tuple[str, int]:
        client = self._ensure_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        last_exc: Exception | None = None
        attempts = max(1, config.LLM_RETRIES + 1)
        for attempt in range(attempts):
            started = time.perf_counter()
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": int(max_tokens or config.LLM_MAX_TOKENS),
                    "temperature": float(
                        config.LLM_TEMPERATURE if temperature is None else temperature
                    ),
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = await asyncio.wait_for(
                    client.chat.completions.create(**kwargs),
                    timeout=self.timeout + 5,
                )
                latency = time.perf_counter() - started
                content = (resp.choices[0].message.content or "").strip()
                usage = getattr(resp, "usage", None)
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                total = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)

                self.usage.calls += 1
                self.usage.ok += 1
                self.usage.prompt_tokens += prompt_tokens
                self.usage.completion_tokens += completion_tokens
                self.usage.last_latency = latency
                return content, total
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - 统一收敛成 LLMError
                last_exc = exc
                self.usage.last_error = f"{type(exc).__name__}: {exc}"
                if attempt + 1 < attempts:
                    await asyncio.sleep(min(2.0 * (attempt + 1), 4.0))

        self.usage.calls += 1
        self.usage.failed += 1
        raise LLMError(f"LLM 调用失败（{attempts} 次重试后）：{last_exc}")

    # ------------------------------------------------------------------ #
    async def complete_text(
        self,
        system: str,
        user: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[str, int]:
        """自由文本 → ``(text, total_tokens)``。"""
        return await self._chat(system, user, max_tokens=max_tokens, temperature=temperature)

    async def complete_json(
        self,
        system: str,
        user: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[dict, int]:
        """要求返回 JSON 对象 → ``(dict, total_tokens)``；顶层不是 dict 直接报错。"""
        text, tokens = await self._chat(
            system, user, max_tokens=max_tokens, temperature=temperature
        )
        data = _extract_json_strict(text, dict)
        return data, tokens

    async def complete_json_list(
        self,
        system: str,
        user: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[list, int]:
        """要 list 用这个（``complete_json`` 只认 dict）。"""
        text, tokens = await self._chat(
            system, user, max_tokens=max_tokens, temperature=temperature
        )
        data = _extract_json_strict(text, list)
        return data, tokens

    def note_degraded(self, reason: str | None = None) -> None:
        """记录一次"该环节本可以调 LLM，但走了规则降级"。"""
        self.usage.degraded += 1
        if reason:
            self.usage.last_error = reason


# --------------------------------------------------------------------------- #
# 进程内单例
# --------------------------------------------------------------------------- #
_client: LLMClient | None = None


def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def set_client(client: LLMClient | None) -> None:
    """测试注入用（假客户端 / 指定端点）。"""
    global _client
    _client = client


__all__ = [
    "LLMClient",
    "LLMError",
    "LLMUnavailable",
    "LLMUsage",
    "get_client",
    "set_client",
    "_extract_json",
]