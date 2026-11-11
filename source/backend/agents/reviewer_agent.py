"""质检 Agent：抽 5 帧做客观指标 + 人眼评分点，判定是否达标，不达标给出下一轮参数。

硬指标（``data/review_rules.json``）：
* ``tex_ratio = |修复区高频纹理方差 − 邻域环带高频方差| / max(邻域高频方差, 4)`` < 0.3
  （对应任务书的「修复区方差 − 邻域方差」；用高频分量是因为整块方差会被背景
  大尺度渐变主导，实测同一块区域在缓变背景上整块方差能差 2~4 倍，无法判定）
* ``0.35 <= edge_ratio <= 1.8``（>1.8 水印边缘残留，<0.35 修糊了）；
  环带自身边缘密度 < ``edge_ring_floor`` 时不适用（纯色/渐变背景），此项豁免
* ``lum_diff`` < 0.9（修复区明显偏色/偏亮视为色块）

每个指标取 5 帧的**中位数**判定，同时要求至少 ``min_frame_pass_ratio`` 的帧
单帧也达标（防止中位数掩盖系统性问题）。``var_ratio`` 保留为诊断值，不做判定。

总分 = 纹理 0.35 + 边缘 0.30 + 色彩 0.20 + 自然度 0.15，>=75 且硬指标全过才算达标。
自然度由规则几何平均给出；有 DeepSeek 时让它写一句人眼评语并给 0-100 的自然度，
两者各占一半 —— LLM 只影响 15% 权重的这一维，**判定不会只由 LLM 决定**。

产物：
* ``review_frames/frame_01..05.jpg``：原片 | 修复后 的并排切片（人工评分点）；
* ``review_frames/montage.jpg``：5 帧汇总；
* ``review_report.json``：由报告 Agent 落盘（本环节把结果写进 payload）。
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import config
import watermark_utils as wu
from agents.base import BaseAgent


class ReviewerAgent(BaseAgent):
    name = "reviewer"
    label = "质检 Agent"
    description = "抽 5 帧核对邻域方差/边缘残留，判断达标并给出回灌参数"

    # ------------------------------------------------------------------ #
    async def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        run_dir = self.run_dir(payload)
        attempt = int(payload.get("attempt", 1))
        rules = self.load_json("review_rules.json")
        thresholds = rules.get("thresholds") or {}
        var_thr = float(thresholds.get("var_ratio", config.QUALITY_VAR_RATIO))
        tex_thr = float(thresholds.get("tex_ratio", var_thr))
        edge_min = float(thresholds.get("edge_ratio_min", 0.35))
        edge_max = float(thresholds.get("edge_ratio_max", 1.8))
        lum_thr = float(thresholds.get("lum_diff", 0.9))
        pass_score = float(thresholds.get("pass_score", 75))
        edge_ring_floor = float(thresholds.get("edge_ring_floor", 0.02))
        min_pass_ratio = float(thresholds.get("min_frame_pass_ratio", 0.6))
        frame_count = int(rules.get("frames", config.REVIEW_FRAMES))

        self.started(payload, f"抽 {frame_count} 帧质检（第 {attempt} 轮）")
        refine = payload.get("refine") or {}
        final_path = Path(refine.get("output") or run_dir / "clean.mp4")
        src_path = Path(payload["src_path"])
        if not final_path.is_file():
            raise RuntimeError(f"找不到成品视频：{final_path}")

        mask = payload.get("mask") or {}
        regions = mask.get("regions") or []
        detect = payload.get("detect") or {}
        video = detect.get("video") or {}

        if not regions:
            review = {
                "passed": True,
                "verdict": "pass",
                "score": 100.0,
                "dimensions": {},
                "thresholds": thresholds,
                "regions": [],
                "comment": "画面中没有达到阈值的静态水印，无需修复，质检直接通过。",
                "frames": [],
                "artifacts": [],
                "attempt": attempt,
                "elapsed": round(time.perf_counter() - started, 3),
            }
            payload["review"] = review
            self.completed(payload, "无区域可检，质检通过", passed=True, score=100)
            return payload

        final_video = await self.to_thread(wu.probe_video, final_path)
        height, width = int(final_video["height"]), int(final_video["width"])
        total = int(final_video["frame_count"] or video.get("frame_count") or 0)
        indices = wu.evenly_spaced_indices(total, frame_count)

        before_frames = await self.to_thread(wu.read_frames_at, src_path, indices)
        after_frames = await self.to_thread(wu.read_frames_at, final_path, indices)
        pairs = list(zip(before_frames, after_frames))
        if not pairs:
            raise RuntimeError("质检抽帧失败：读不到成片帧")

        ring_masks = wu._region_ring_masks((height, width), regions)

        # ---------------- 逐帧客观指标 ---------------- #
        per_frame: list[dict[str, Any]] = []
        for frame_idx, (before, after) in zip(indices, pairs):
            if after.shape[:2] != (height, width):
                after = cv2.resize(after, (width, height))
            gray_after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
            metrics = wu.measure_frame(gray_after, ring_masks, regions)
            per_frame.append({"index": int(frame_idx), "metrics": metrics})

        # ---------------- 落盘 5 个人工评分点 ---------------- #
        artifacts = await self._save_review_images(
            run_dir, regions, pairs, indices, per_frame, src_path.name, attempt
        )

        # ---------------- 汇总与评分 ---------------- #
        agg, regions_agg = self._aggregate(
            regions, per_frame, tex_thr, edge_min, edge_max, edge_ring_floor
        )
        dimensions = self._score_dimensions(
            agg, regions_agg, tex_thr, edge_min, edge_max, lum_thr
        )

        llm_comment, llm_natural, llm_plan = await self._llm_comment(
            payload, agg, regions_agg, dimensions, thresholds, attempt
        )
        if llm_natural is not None:
            rule_natural = dimensions["natural"]["score"]
            dimensions["natural"]["rule_score"] = rule_natural
            dimensions["natural"]["llm_score"] = round(llm_natural, 1)
            dimensions["natural"]["score"] = round(0.5 * rule_natural + 0.5 * llm_natural, 1)

        total_score = round(
            sum(d["score"] * d["weight"] for d in dimensions.values()), 2
        )

        failures: list[str] = []
        for item in regions_agg:
            if item["tex_ratio"] > tex_thr:
                failures.append(
                    f"{item['id']} 纹理不接续：纹理方差差 {item['tex_ratio']:.2f} > {tex_thr}"
                )
            elif item["frame_pass_ratio"] < min_pass_ratio:
                failures.append(
                    f"{item['id']} 指标抖动过大：仅 {item['pass_frames']}/{item['samples']} 帧达标"
                    f"（要求 ≥{min_pass_ratio:.0%}）"
                )
            if not item["edge_exempt"]:
                if item["edge_ratio"] > edge_max:
                    failures.append(
                        f"{item['id']} 水印边缘残留：边缘密度比 {item['edge_ratio']:.2f} > {edge_max}"
                    )
                elif item["edge_ratio"] < edge_min:
                    failures.append(
                        f"{item['id']} 修复过度平滑：边缘密度比 {item['edge_ratio']:.2f} < {edge_min}"
                    )
            if item["lum_diff"] > lum_thr:
                failures.append(
                    f"{item['id']} 亮度/颜色不连续：{item['lum_diff']:.2f} > {lum_thr}"
                )
        if total_score < pass_score:
            failures.append(f"综合得分 {total_score} < {pass_score}")

        passed = not failures
        verdict = "pass" if passed else "fail"

        # ---------------- 不达标：给出下一轮参数 ---------------- #
        next_params = None
        next_overrides: dict[str, dict[str, Any]] = {}
        if not passed:
            grid = self.load_json("inpaint_grid.json")
            next_params, next_overrides = self._next_plan(
                payload, grid, regions_agg, thresholds, llm_plan
            )

        comment = llm_comment or self._rule_comment(passed, failures, regions_agg)
        review = {
            "passed": passed,
            "verdict": verdict,
            "score": total_score,
            "pass_score": pass_score,
            "dimensions": dimensions,
            "thresholds": thresholds,
            "regions": regions_agg,
            "frames": per_frame,
            "frame_indices": indices,
            "failures": failures,
            "comment": comment,
            "next_params": next_params,
            "next_overrides": next_overrides,
            "attempt": attempt,
            "artifacts": artifacts,
            "elapsed": round(time.perf_counter() - started, 3),
        }
        payload["review"] = review
        payload["last_failure"] = "; ".join(failures) if failures else None

        payload["trace"] = payload.get("trace", [])
        payload["trace"].append(
            {
                "agent": self.name,
                "seconds": review["elapsed"],
                "passed": passed,
                "score": total_score,
                "attempt": attempt,
            }
        )

        if passed:
            message = f"质检通过：综合 {total_score} 分，硬指标全部达标"
        else:
            message = f"质检不达标（{total_score} 分）：" + "；".join(failures[:3])
        self.completed(
            payload,
            message,
            passed=passed,
            score=total_score,
            failures=len(failures),
        )
        return payload

    # ------------------------------------------------------------------ #
    # 评分
    # ------------------------------------------------------------------ #
    @staticmethod
    def _aggregate(
        regions: list[dict[str, Any]],
        per_frame: list[dict[str, Any]],
        tex_thr: float,
        edge_min: float,
        edge_max: float,
        edge_ring_floor: float,
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """逐区域把 5 帧指标汇总成判定用的「中位数 + 逐帧通过率」。

        单帧指标在运动物体穿过环带时会剧烈抖动（实测同一区域 tex_ratio 可在
        0.01~0.6 之间跳），逐帧全过会把"整体已经修好"的结果判成不达标；
        因此判定用中位数，同时要求多数帧也单帧达标（防止中位数掩盖系统性问题）。
        """
        regions_agg: list[dict[str, Any]] = []
        for i, region in enumerate(regions):
            rows = [
                frame["metrics"][i]
                for frame in per_frame
                if i < len(frame["metrics"]) and frame["metrics"][i]
            ]
            if not rows:
                regions_agg.append(
                    {"id": region.get("id"), "samples": 0, "tex_ratio": 0.0,
                     "var_ratio": 0.0, "edge_ratio": 1.0, "lum_diff": 0.0,
                     "edge_exempt": True, "frame_pass_ratio": 1.0, "pass_frames": 0}
                )
                continue

            tex_series = np.array([r["tex_ratio"] for r in rows], dtype=float)
            var_series = np.array([r["var_ratio"] for r in rows], dtype=float)
            edge_series = np.array([r["edge_ratio"] for r in rows], dtype=float)
            lum_series = np.array([r["lum_diff"] for r in rows], dtype=float)
            edge_ring_med = float(np.median([r.get("edge_ring", 0.0) for r in rows]))
            # 环带本身几乎没有边缘（纯色/渐变背景）时，边缘密度比没有意义 → 豁免
            edge_exempt = edge_ring_med < edge_ring_floor
            edge_ok = edge_exempt | ((edge_series >= edge_min) & (edge_series <= edge_max))
            frame_pass = (tex_series <= tex_thr) & edge_ok & (lum_series <= 0.9)
            pass_frames = int(frame_pass.sum())

            regions_agg.append(
                {
                    "id": region.get("id"),
                    "bbox": region.get("bbox"),
                    "samples": len(rows),
                    "tex_ratio": round(float(np.median(tex_series)), 4),
                    "tex_ratio_mean": round(float(tex_series.mean()), 4),
                    "tex_ratio_max": round(float(tex_series.max()), 4),
                    "tex_ratio_min": round(float(tex_series.min()), 4),
                    "tex_in": round(float(np.median([r.get("tex_in", 0.0) for r in rows])), 2),
                    "tex_ring": round(float(np.median([r.get("tex_ring", 0.0) for r in rows])), 2),
                    "var_ratio": round(float(np.median(var_series)), 4),
                    "var_ratio_max": round(float(var_series.max()), 4),
                    "edge_ratio": round(float(np.median(edge_series)), 4),
                    "edge_ratio_mean": round(float(edge_series.mean()), 4),
                    "edge_ratio_max": round(float(edge_series.max()), 4),
                    "edge_ratio_min": round(float(edge_series.min()), 4),
                    "edge_ring": round(edge_ring_med, 5),
                    "edge_exempt": bool(edge_exempt),
                    "lum_diff": round(float(np.median(lum_series)), 4),
                    "var_in": round(float(np.median([r["var_in"] for r in rows])), 2),
                    "var_ring": round(float(np.median([r["var_ring"] for r in rows])), 2),
                    "pass_frames": pass_frames,
                    "frame_pass_ratio": round(pass_frames / max(len(rows), 1), 3),
                }
            )
        agg = {
            "tex_ratio": round(float(np.median([r["tex_ratio"] for r in regions_agg])), 4),
            "var_ratio": round(float(np.median([r["var_ratio"] for r in regions_agg])), 4),
            "edge_ratio": round(float(np.median([r["edge_ratio"] for r in regions_agg])), 4),
            "lum_diff": round(float(np.median([r["lum_diff"] for r in regions_agg])), 4),
        }
        return agg, regions_agg

    @staticmethod
    def _score_dimensions(
        agg: dict[str, float],
        regions_agg: list[dict[str, Any]],
        tex_thr: float,
        edge_min: float,
        edge_max: float,
        lum_thr: float,
    ) -> dict[str, dict[str, Any]]:
        def clamp01(v: float) -> float:
            return max(0.0, min(1.0, float(v)))

        tex_ratio = max((r["tex_ratio"] for r in regions_agg), default=0.0)
        texture = 100.0 * clamp01(1.0 - tex_ratio / max(tex_thr, 1e-6))

        # 边缘：区间内满分，偏高按 0.6*hi 的尺度惩罚，偏低按 lo 的尺度惩罚；
        # 环带本身没有边缘时（纯色背景）这项不适用，直接给满分
        edge_ratios = [
            r["edge_ratio"] for r in regions_agg if not r.get("edge_exempt")
        ] or [1.0]
        edge_scores = []
        for ratio in edge_ratios:
            if edge_min <= ratio <= edge_max:
                edge_scores.append(100.0)
            elif ratio > edge_max:
                edge_scores.append(100.0 * clamp01(1.0 - (ratio - edge_max) / (0.6 * edge_max)))
            else:
                edge_scores.append(100.0 * clamp01(ratio / max(edge_min, 1e-6)))
        edge = float(np.mean(edge_scores)) if edge_scores else 100.0

        lum = max((r["lum_diff"] for r in regions_agg), default=0.0)
        color = 100.0 * clamp01(1.0 - lum / max(lum_thr, 1e-6))

        geo = max(1.0, (max(texture, 1.0) * max(edge, 1.0) * max(color, 1.0)) ** (1.0 / 3.0))
        return {
            "texture": {"label": "纹理一致性", "score": round(texture, 2), "weight": 0.35,
                        "detail": f"最差区域纹理方差差 {tex_ratio:.3f}（阈值 {tex_thr}）"},
            "edge": {"label": "边缘残留", "score": round(edge, 2), "weight": 0.30,
                     "detail": f"边缘密度比 {min(edge_ratios):.2f}~{max(edge_ratios):.2f}"
                               f"（合格区间 {edge_min}~{edge_max}）"},
            "color": {"label": "色彩/亮度连续性", "score": round(color, 2), "weight": 0.20,
                      "detail": f"最大亮度差 {lum:.3f}（阈值 {lum_thr}）"},
            "natural": {"label": "整体自然度", "score": round(geo, 2), "weight": 0.15,
                        "detail": "规则：三维几何平均；有 LLM 时与模型评分各占一半"},
        }

    @staticmethod
    def _rule_comment(passed: bool, failures: list[str], regions_agg: list[dict[str, Any]]) -> str:
        if passed:
            worst = max(regions_agg, key=lambda r: r["tex_ratio"]) if regions_agg else None
            if worst:
                return (
                    f"修复区纹理与邻域一致（最差纹理方差差 {worst['tex_ratio']:.2f}），"
                    f"边缘密度比 {worst['edge_ratio']:.2f}"
                    f"{'（背景无边缘，已豁免）' if worst.get('edge_exempt') else ' 在合格区间内'}，"
                    f"肉眼看不出修痕。"
                )
            return "未发现残留水印，修复结果自然。"
        head = failures[0] if failures else "指标未达标"
        return f"未达标：{head}；建议按下一轮参数重生成。"

    # ------------------------------------------------------------------ #
    # 图片落盘
    # ------------------------------------------------------------------ #
    async def _save_review_images(
        self,
        run_dir: Path,
        regions: list[dict[str, Any]],
        pairs: list[tuple[np.ndarray, np.ndarray]],
        indices: list[int],
        per_frame: list[dict[str, Any]],
        src_name: str,
        attempt: int,
    ) -> list[str]:
        frame_dir = run_dir / "review_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[str] = []
        tiles: list[np.ndarray] = []

        for n, ((before, after), idx, frame) in enumerate(zip(pairs, indices, per_frame), start=1):
            panels: list[np.ndarray] = []
            labels: list[str] = []
            for region in regions:
                box = region["bbox"]
                panels.append(wu.crop_with_context(before, box))
                panels.append(wu.crop_with_context(after, box))
                labels.extend([f"原片 {region['id']}", f"修复后 {region['id']}"])
            tile = wu.side_by_side(panels, labels, tile_height=180) if panels else before
            tile = tile.copy()
            header = np.full((28, tile.shape[1], 3), 18, np.uint8)
            cv2.putText(
                header,
                f"第 {n}/5 个人工评分点  frame={idx}  第{attempt}轮  attempt 综合分={_frame_score(frame)}",
                (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            tile = np.vstack([header, tile])
            path = frame_dir / f"frame_{n:02d}.jpg"
            await self.to_thread(wu.write_image, path, tile)
            artifacts.append(str(path))
            tiles.append(tile)

        if tiles:
            width = max(t.shape[1] for t in tiles)
            padded = []
            for tile in tiles:
                if tile.shape[1] < width:
                    pad = np.full((tile.shape[0], width - tile.shape[1], 3), 18, np.uint8)
                    tile = np.hstack([tile, pad])
                padded.append(tile)
            montage = np.vstack(padded)
            montage_path = frame_dir / "montage.jpg"
            await self.to_thread(wu.write_image, montage_path, montage)
            artifacts.append(str(montage_path))
        return artifacts

    # ------------------------------------------------------------------ #
    # LLM 评语
    # ------------------------------------------------------------------ #
    async def _llm_comment(
        self,
        payload: dict[str, Any],
        agg: dict[str, float],
        regions_agg: list[dict[str, Any]],
        dimensions: dict[str, dict[str, Any]],
        thresholds: dict[str, Any],
        attempt: int,
    ) -> tuple[str | None, float | None, dict[str, Any] | None]:
        system = self.read_prompt("reviewer_system.txt")
        if not system:
            return None, None, None
        view = [
            {
                "id": r["id"],
                "tex_ratio": r["tex_ratio"],
                "var_ratio": r["var_ratio"],
                "edge_ratio": r["edge_ratio"],
                "edge_exempt": r["edge_exempt"],
                "lum_diff": r["lum_diff"],
            }
            for r in regions_agg
        ]
        user = (
            f"第 {attempt} 轮修复后的质检数据如下（全部为实测值）。\n"
            f"阈值：{thresholds}\n"
            f"逐区域指标（JSON）：{view}\n"
            f"维度得分：{ {k: v['score'] for k, v in dimensions.items()} }\n"
            "请按系统提示输出 JSON：comment（40 字内）、natural_score（0-100）、"
            "verdict（pass/fail）、next_params（不达标时才需要）、per_region（逐区域建议数组）。"
        )

        def _parse(data: Any) -> dict[str, Any]:
            if not isinstance(data, dict):
                raise ValueError("顶层不是 dict")
            return data

        result, _tokens = await self.call_llm(
            payload, "reviewer_comment", system, user, parse=_parse, json_mode=True
        )
        if not isinstance(result, dict):
            return None, None, None
        comment = str(result.get("comment", "")).strip()[:120] or None
        natural: float | None = None
        try:
            raw = float(result.get("natural_score"))
            if not math.isnan(raw):
                natural = max(0.0, min(100.0, raw))
        except (TypeError, ValueError):
            natural = None
        plan = result.get("next_params")
        return comment, natural, plan if isinstance(plan, dict) else None

    # ------------------------------------------------------------------ #
    def _next_plan(
        self,
        payload: dict[str, Any],
        grid: dict[str, Any],
        regions_agg: list[dict[str, Any]],
        thresholds: dict[str, Any],
        llm_plan: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        """下一轮参数：梯度表下一档 + 逐区域 mask 覆盖 + LLM 建议（严格校验后合入）。"""
        steps = grid.get("grid") or []
        attempt = int(payload.get("attempt", 1))
        base = dict(steps[min(attempt, len(steps) - 1)]) if steps else {}
        base.pop("why", None)
        base.pop("attempt", None)

        # 逐区域：超标区域给对应覆盖
        tex_thr = float(thresholds.get("tex_ratio", 0.3))
        edge_max = float(thresholds.get("edge_ratio_max", 1.8))
        edge_min = float(thresholds.get("edge_ratio_min", 0.35))
        overrides: dict[str, dict[str, Any]] = {}
        rules = grid.get("region_overrides") or {}
        for item in regions_agg:
            cls = ""
            if item.get("edge_exempt"):
                # 背景无边缘纹理，只能用纹理方差判断（修糊了方差会塌）
                if item["tex_ratio"] > tex_thr:
                    cls = "edge_residual" if item["tex_ratio"] < 0.6 else "over_smooth"
            elif item["edge_ratio"] > edge_max or item["tex_ratio"] > tex_thr:
                cls = "edge_residual"
            elif item["edge_ratio"] < edge_min:
                cls = "over_smooth"
            if cls and cls in rules:
                overrides[item["id"]] = {"class": cls, **{k: v for k, v in rules[cls].items() if k != "why"}}

        # LLM 建议只在允许集合内生效
        if isinstance(llm_plan, dict):
            allowed = {
                "method": grid.get("allowed_methods") or [],
                "radius": grid.get("allowed_radius") or [],
                "blur_mix": grid.get("allowed_blur_mix") or [],
                "feather_sigma": grid.get("allowed_feather_sigma") or [],
                "dilate_extra": grid.get("allowed_dilate_extra") or [],
                "ink_threshold": grid.get("allowed_ink_threshold") or [],
                "pad_ratio": grid.get("allowed_pad_ratio") or [],
            }
            for key, values in allowed.items():
                if key not in llm_plan or not values:
                    continue
                try:
                    if key == "method":
                        if str(llm_plan[key]).lower() in values:
                            base[key] = str(llm_plan[key]).lower()
                    else:
                        value = float(llm_plan[key])
                        if value in [float(v) for v in values]:
                            base[key] = value
                except (TypeError, ValueError):
                    continue
            base["llm_adjusted"] = True
        return base, overrides


def _frame_score(frame: dict[str, Any]) -> str:
    """给图上标注一个粗略的帧级指标（方差差最大值）。"""
    metrics = frame.get("metrics") or []
    if not metrics:
        return "n/a"
    worst = max((m.get("var_ratio", 0.0) for m in metrics), default=0.0)
    return f"varΔ={worst:.2f}"