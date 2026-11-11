"""报告 Agent（并行节点）：把整条流水线的结果落成两份可审计的 JSON。

``detect_report.json``：视频信息、抽样参数、全部候选（bbox + 置信度 + 各项子分数）、
最终选区、mask 参数、修复参数与逐帧统计、精修记录、质检结论、LLM 用量、链路耗时。

``review_report.json``：质检维度得分、逐区域指标、失败原因、评语、回灌参数、
人工评分点图片路径。

两份报告都只从 payload 里已有的字段取数，不重新测、不编造。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import config
from agents.base import BaseAgent


class ReportAgent(BaseAgent):
    name = "report"
    label = "报告 Agent"
    description = "落盘 detect_report.json / review_report.json，审计可追溯"

    async def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        run_dir = self.run_dir(payload)
        self.started(payload, "汇总检测/质检报告")
        run_id = self.run_id(payload)

        detect = payload.get("detect") or {}
        mask = payload.get("mask") or {}
        repair = payload.get("repair") or {}
        refine = payload.get("refine") or {}
        review = payload.get("review") or {}
        preview = payload.get("preview") or {}
        llm_status = self.llm.status
        video = detect.get("video") or {}

        def rel(paths: list[str]) -> list[str]:
            out = []
            for p in paths:
                try:
                    out.append(str(Path(p).resolve().relative_to(run_dir.resolve())).replace("\\", "/"))
                except (ValueError, OSError):
                    out.append(str(p))
            return out

        detect_artifacts = rel((detect.get("artifacts") or []) + (mask.get("artifacts") or []) + (preview.get("artifacts") or []))
        review_artifacts = rel(review.get("artifacts") or [])

        detect_report = {
            "run_id": run_id,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "app": {"name": config.APP_NAME, "version": config.APP_VERSION},
            "request": payload.get("params") or {},
            "authorized_note": "用户已勾选「仅处理本人拥有版权或已获授权的内容」",
            "video": video,
            "detection": {
                "method": "多帧投票（Canny 边缘出现率 > 阈值）",
                "sample": detect.get("sample"),
                "appear_rate_threshold": config.DETECT_APPEAR_RATE,
                "candidates": detect.get("candidates") or [],
                "selected": detect.get("selected") or [],
                "regions": detect.get("regions") or [],
                "review_source": detect.get("review_source"),
                "policy": detect.get("policy") or {},
                "elapsed": detect.get("elapsed"),
            },
            "mask": {
                "coverage": mask.get("coverage"),
                "regions": mask.get("regions") or [],
                "plan": mask.get("plan") or {},
                "feather_note": "羽化：cv2.GaussianBlur(binary, (0,0), sigma=5~10)，回贴时做 alpha 软合成",
            },
            "repair": {
                "params": repair.get("params") or {},
                "stats": repair.get("stats") or {},
                "elapsed": repair.get("elapsed"),
            },
            "refine": {
                "applied": refine.get("applied"),
                "params": refine.get("params") or {},
                "problems": refine.get("problems") or {},
                "reason": refine.get("reason"),
                "elapsed": refine.get("elapsed"),
            },
            "review_summary": {
                "passed": review.get("passed"),
                "score": review.get("score"),
                "attempt": review.get("attempt"),
                "failures": review.get("failures") or [],
                "comment": review.get("comment"),
            },
            "llm": llm_status,
            "trace": payload.get("trace") or [],
            "artifacts": detect_artifacts,
            "outputs": {
                "clean_video": "clean.mp4",
                "detect_report": "detect_report.json",
                "review_report": "review_report.json",
            },
        }
        detect_path = run_dir / "detect_report.json"
        await self.to_thread(self.write_json, detect_path, detect_report)

        review_report = {
            "run_id": run_id,
            "generated_at": detect_report["generated_at"],
            "passed": review.get("passed"),
            "verdict": review.get("verdict"),
            "score": review.get("score"),
            "pass_score": review.get("pass_score"),
            "attempts": payload.get("attempt"),
            "feedback_count": payload.get("feedback_count", 0),
            "thresholds": review.get("thresholds") or {},
            "dimensions": review.get("dimensions") or {},
            "regions": review.get("regions") or [],
            "frames": review.get("frames") or [],
            "frame_indices": review.get("frame_indices") or [],
            "failures": review.get("failures") or [],
            "comment": review.get("comment"),
            "next_params": review.get("next_params"),
            "next_overrides": review.get("next_overrides"),
            "human_check": {
                "instruction": "逐张查看 review_frames/frame_0X.jpg：左为原片切片，右为修复后切片",
                "images": review_artifacts,
                "montage": next((p for p in review_artifacts if p.endswith("montage.jpg")), None),
            },
            "llm": llm_status,
            "outputs": {"clean_video": "clean.mp4"},
        }
        review_path = run_dir / "review_report.json"
        await self.to_thread(self.write_json, review_path, review_report)

        # 运行日志：一份人可读的流水线时间线（便于核对"不达标→重生成→达标"）
        run_log = {
            "run_id": run_id,
            "trace": payload.get("trace") or [],
            "events": [
                {
                    "attempt": review.get("attempt"),
                    "passed": review.get("passed"),
                    "score": review.get("score"),
                    "failures": review.get("failures") or [],
                }
            ],
            "llm": llm_status,
            "config": {
                "detect_sample_frames": config.DETECT_SAMPLE_FRAMES,
                "detect_work_width": config.DETECT_WORK_WIDTH,
                "appear_rate": config.DETECT_APPEAR_RATE,
                "max_feedback": config.MAX_FEEDBACK,
                "quality_var_ratio": config.QUALITY_VAR_RATIO,
            },
        }
        run_log_path = run_dir / "run_log.json"
        await self.to_thread(self.write_json, run_log_path, run_log)

        payload["report"] = {
            "detect_report": str(detect_path),
            "review_report": str(review_path),
            "run_log": str(run_log_path),
            "artifacts": detect_artifacts + review_artifacts,
            "elapsed": round(time.perf_counter() - started, 3),
        }
        payload["trace"] = payload.get("trace", [])
        payload["trace"].append(
            {"agent": self.name, "seconds": payload["report"]["elapsed"], "reports": 3}
        )
        self.completed(
            payload,
            "报告已落盘：detect_report.json / review_report.json / run_log.json",
            reports=3,
        )
        return payload