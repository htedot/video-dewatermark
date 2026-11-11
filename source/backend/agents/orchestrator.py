"""Orchestrator：注册 Agent、按 DAG 分层调度、处理质检回灌。

DAG（强依赖串行，末尾一小组并行聚合）：

    detector → mask → inpainter → refiner → reviewer → ┬ preview ┐
                       ↑                                └ report  ┘
                       └──── feedback: reviewer 不达标 → 回到 mask ────┘

* 每层执行完 emit ``agent_started/completed/failed``；
* 并行层用 ``asyncio.gather`` 同时跑 preview 与 report，前后 emit
  ``parallel_started`` / ``parallel_completed``（聚合点）；
* reviewer 判不达标时，把 ``next_params`` / ``next_overrides`` 写回 payload，
  从 mask 层重新跑一遍（最多 ``MAX_FEEDBACK`` 次），并 emit ``review_feedback``；
* 所有 Agent 只交换 payload，Orchestrator 不 import 任何业务算法。

Agent 的 ``run`` 既可以是协程也可以是同步函数：协程直接 await，同步函数丢进
线程池（不阻塞事件循环，SSE 才能持续推）。
"""

from __future__ import annotations

import asyncio
import inspect
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import config
import llm as llm_module
from agents.base import BaseAgent
from agents.detector_agent import DetectorAgent
from agents.inpainter_agent import InpainterAgent
from agents.mask_agent import MaskAgent
from agents.preview_agent import PreviewAgent
from agents.refiner_agent import RefinerAgent
from agents.report_agent import ReportAgent
from agents.reviewer_agent import ReviewerAgent
from event_bus import EventBus


@dataclass
class Node:
    key: str
    label: str
    deps: tuple[str, ...] = ()
    parallel_group: str | None = None


class Orchestrator:
    """基于分层 DAG 的调度器（层级在构造时算好，运行时只做顺序执行 + 回灌）。"""

    def __init__(
        self,
        bus: EventBus,
        llm: llm_module.LLMClient | None = None,
        max_feedback: int | None = None,
    ) -> None:
        self.bus = bus
        self.llm = llm or llm_module.get_client()
        self.max_feedback = int(
            config.MAX_FEEDBACK if max_feedback is None else max_feedback
        )
        self.agents: dict[str, BaseAgent] = {
            "detector": DetectorAgent(bus, self.llm),
            "mask": MaskAgent(bus, self.llm),
            "inpainter": InpainterAgent(bus, self.llm),
            "refiner": RefinerAgent(bus, self.llm),
            "reviewer": ReviewerAgent(bus, self.llm),
            "preview": PreviewAgent(bus, self.llm),
            "report": ReportAgent(bus, self.llm),
        }
        self.nodes: list[Node] = [
            Node("detector", "检测", ()),
            Node("mask", "遮罩", ("detector",)),
            Node("inpainter", "修复", ("mask",)),
            Node("refiner", "精修", ("inpainter",)),
            Node("reviewer", "质检", ("refiner",)),
            Node("preview", "对比预览", ("reviewer",), parallel_group="reporting"),
            Node("report", "报告", ("reviewer",), parallel_group="reporting"),
        ]
        self.layers: list[list[str]] = self._compute_layers()
        self.feedback_target = "mask"
        self.active: set[str] = set()

    # ------------------------------------------------------------------ #
    # DAG 描述
    # ------------------------------------------------------------------ #
    def _compute_layers(self) -> list[list[str]]:
        by_key = {node.key: node for node in self.nodes}
        layers: list[list[str]] = []
        remaining = set(by_key)
        while remaining:
            layer = [
                key
                for key in (node.key for node in self.nodes)
                if key in remaining and all(dep not in remaining for dep in by_key[key].deps)
            ]
            if not layer:
                raise RuntimeError(f"DAG 有环：{sorted(remaining)}")
            layers.append(layer)
            remaining -= set(layer)
        return layers

    def describe(self) -> dict[str, Any]:
        return {
            "layers": self.layers,
            "nodes": [
                {
                    "key": node.key,
                    "label": node.label,
                    "deps": list(node.deps),
                    "parallel_group": node.parallel_group,
                    "description": self.agents[node.key].description,
                }
                for node in self.nodes
            ],
            "feedback": {"from": "reviewer", "to": self.feedback_target,
                         "max_retries": self.max_feedback},
            "max_feedback": self.max_feedback,
        }

    # ------------------------------------------------------------------ #
    # 执行
    # ------------------------------------------------------------------ #
    async def _run_node(self, key: str, payload: dict[str, Any]) -> dict[str, Any]:
        agent = self.agents[key]
        started = time.perf_counter()
        try:
            if inspect.iscoroutinefunction(agent.run):
                result = await agent.run(payload)
            else:  # 同步 Agent 丢线程池，别卡住 SSE
                result = await asyncio.to_thread(agent.run, payload)
            if isinstance(result, dict):
                payload = result
            return payload
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - 统一上报
            elapsed = round(time.perf_counter() - started, 3)
            self.bus.emit(
                "agent_failed",
                payload.get("run_id"),
                f"{agent.label} 失败：{type(exc).__name__}: {exc}",
                agent=key,
                label=agent.label,
                elapsed=elapsed,
                error=str(exc),
                traceback=traceback.format_exc()[-1500:],
            )
            raise

    async def _run_layer(self, layer: list[str], payload: dict[str, Any]) -> None:
        if len(layer) == 1:
            await self._run_node(layer[0], payload)
            return
        self.bus.emit(
            "parallel_started",
            payload.get("run_id"),
            f"并行层开始：{' + '.join(self.agents[k].label for k in layer)}",
            nodes=layer,
        )
        started = time.perf_counter()
        await asyncio.gather(*(self._run_node(key, payload) for key in layer))
        self.bus.emit(
            "parallel_completed",
            payload.get("run_id"),
            f"并行层完成：{' + '.join(self.agents[k].label for k in layer)}",
            nodes=layer,
            elapsed=round(time.perf_counter() - started, 3),
        )

    def _layer_index(self, key: str) -> int:
        for i, layer in enumerate(self.layers):
            if key in layer:
                return i
        return 0

    def _check_feedback(self, payload: dict[str, Any]) -> bool:
        """reviewer 不达标且还有重试额度 → 准备下一轮参数并返回 True。"""
        review = payload.get("review") or {}
        if not review or review.get("passed", True):
            return False
        count = int(payload.get("feedback_count", 0))
        if count >= self.max_feedback:
            self.bus.emit(
                "run_log",
                payload.get("run_id"),
                f"已回灌 {count} 次仍未达标，按最后一轮结果交付（报告里会注明）",
                level="warning",
            )
            return False
        payload["feedback_count"] = count + 1
        payload["retry_plan"] = review.get("next_params") or {}
        payload["retry_overrides"] = review.get("next_overrides") or {}
        payload["attempt"] = int(payload.get("attempt", 1)) + 1
        # 每轮回灌单独一个阶段目录，避免多轮产物互相覆盖（自测时踩过：三档参数
        # 全写进 attempt_1/，事后看不出哪一轮是哪个结果）
        run_dir = Path(payload.get("run_dir") or (config.OUTPUT_DIR / str(payload.get("run_id"))))
        stage_dir = run_dir / f"attempt_{payload['attempt']}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        payload["stage_dir"] = str(stage_dir)
        self.bus.emit(
            "review_feedback",
            payload.get("run_id"),
            f"质检不达标 → 回灌第 {payload['attempt']} 轮："
            + "；".join((review.get("failures") or [])[:3]),
            attempt=payload["attempt"],
            failures=review.get("failures") or [],
            next_params=payload.get("retry_plan"),
            overrides=payload.get("retry_overrides"),
        )
        return True

    async def _execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        index = 0
        while index < len(self.layers):
            layer = self.layers[index]
            await self._run_layer(layer, payload)
            if index == self._layer_index("reviewer") and self._check_feedback(payload):
                index = self._layer_index(self.feedback_target)
                continue
            index += 1
        return payload

    # ------------------------------------------------------------------ #
    # 对外入口
    # ------------------------------------------------------------------ #
    async def run_job(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """跑完一条完整流水线；任何阶段失败都会 emit task_failed，不会静默。"""
        self.active.add(run_id)
        payload.setdefault("run_id", run_id)
        payload.setdefault("attempt", 1)
        payload.setdefault("feedback_count", 0)
        payload.setdefault("trace", [])
        payload.setdefault("run_dir", str(config.OUTPUT_DIR / run_id))
        run_dir = Path(payload["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        payload.setdefault("stage_dir", str(run_dir / f"attempt_{payload['attempt']}"))

        started = time.perf_counter()
        self.bus.emit(
            "task_started",
            run_id,
            "流水线启动：检测 → 遮罩 → 修复 → 精修 → 质检 → 预览/报告",
            dag=self.layers,
            src=payload.get("src_path"),
        )
        status = "completed"
        try:
            await asyncio.wait_for(self._execute(payload), timeout=config.JOB_TIMEOUT)
            review = payload.get("review") or {}
            artifacts = self._collect_artifacts(run_dir)
            payload["artifacts"] = artifacts
            payload["elapsed"] = round(time.perf_counter() - started, 2)
            self.bus.emit(
                "task_completed",
                run_id,
                (
                    f"任务完成：质检{'通过' if review.get('passed') else '未通过'}"
                    f"（{review.get('score')} 分，共 {payload.get('attempt')} 轮），"
                    f"产物 {len(artifacts)} 个"
                ),
                passed=bool(review.get("passed")),
                score=review.get("score"),
                attempts=payload.get("attempt"),
                feedback_count=payload.get("feedback_count", 0),
                elapsed=payload["elapsed"],
                artifacts=artifacts,
            )
        except asyncio.TimeoutError:
            status = "failed"
            payload["error"] = f"任务超时（>{config.JOB_TIMEOUT}s）"
            self.bus.emit("task_failed", run_id, payload["error"], error=payload["error"])
        except asyncio.CancelledError:
            status = "cancelled"
            self.bus.emit("task_failed", run_id, "任务被取消", error="cancelled")
            raise
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            payload["error"] = f"{type(exc).__name__}: {exc}"
            self.bus.emit(
                "task_failed",
                run_id,
                f"任务失败：{payload['error']}",
                error=payload["error"],
                traceback=traceback.format_exc()[-2000:],
            )
        finally:
            self.active.discard(run_id)
            payload["status"] = status
        return payload

    @staticmethod
    def _collect_artifacts(run_dir: Path) -> list[dict[str, Any]]:
        """按相对路径列出产物（size 便于前端直接显示/下载）。"""
        out: list[dict[str, Any]] = []
        if not run_dir.is_dir():
            return out
        for path in sorted(run_dir.rglob("*")):
            if path.is_file() and path.name != "source_path.txt":
                out.append(
                    {
                        "name": str(path.relative_to(run_dir)).replace("\\", "/"),
                        "bytes": path.stat().st_size,
                    }
                )
        return out


__all__ = ["Node", "Orchestrator"]