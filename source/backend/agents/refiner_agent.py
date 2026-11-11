"""精修 Agent：拿修复环节的逐帧指标，逐区域判断"哪块还没修好"，再针对性重修一次。

判定（阈值来自 ``data/review_rules.json``）：
* ``edge_ratio`` 明显偏高 → **边缘残留**：mask 没盖住半透明晕边，或 inpaint 半径不够；
* ``edge_ratio`` 明显偏低 → **过度平滑**：把邻域纹理也抹了，需要收缩 mask、减小半径；
* ``var_ratio`` 超标 → 纹理不接续（按边缘残留处理，加厚 + 换 NS）。

有问题时：从**原片**重新生成一版 mask（问题区域用 ``inpaint_grid.json`` 里
``region_overrides`` 的参数单独加厚/收缩），换参数重新跑一遍 inpaint，
成品写到 ``outputs/<run_id>/clean.mp4``。没问题时直接复用修复结果（不重编码）。
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import config
import watermark_utils as wu
from agents.base import BaseAgent


class RefinerAgent(BaseAgent):
    name = "refiner"
    label = "精修 Agent"
    description = "对比修复区与邻域方差/边缘，超阈值的区域单独重修"

    async def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        run_dir = self.run_dir(payload)
        stage_dir = Path(payload.get("stage_dir") or run_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)
        src = Path(payload["src_path"])
        attempt = int(payload.get("attempt", 1))
        self.started(payload, f"核对修复区方差/边缘密度（第 {attempt} 轮）")

        rules = self.load_json("review_rules.json")
        thresholds = rules.get("thresholds") or {}
        var_thr = float(thresholds.get("var_ratio", config.QUALITY_VAR_RATIO))
        edge_min = float(thresholds.get("edge_ratio_min", 0.35))
        edge_max = float(thresholds.get("edge_ratio_max", config.QUALITY_EDGE_RATIO))

        mask = payload.get("mask") or {}
        regions = mask.get("regions") or []
        repair = payload.get("repair") or {}
        repair_out = Path(repair.get("output") or stage_dir / "inpainted.mp4")
        stats = (repair.get("stats") or {}).get("regions") or []

        if not regions:
            await self._passthrough(payload, repair_out, run_dir, stage_dir, started, "无区域")
            return payload

        problems: dict[str, str] = {}
        details: list[dict[str, Any]] = []
        for item in stats:
            rid = item.get("id")
            if not rid or not item.get("samples"):
                continue
            var_avg = float(item.get("var_ratio_avg", 0.0))
            edge_avg = float(item.get("edge_ratio_avg", 1.0))
            cls = ""
            if edge_avg > edge_max or var_avg > var_thr:
                cls = "edge_residual"
            elif edge_avg < edge_min:
                cls = "over_smooth"
            details.append(
                {
                    "id": rid,
                    "var_ratio": round(var_avg, 4),
                    "edge_ratio": round(edge_avg, 4),
                    "verdict": cls or "ok",
                }
            )
            if cls:
                problems[rid] = cls

        self.progress(
            payload,
            f"逐区域核对：{len(details)} 个区域，{len(problems)} 个需要重修",
            percent=25,
            problems=len(problems),
        )

        if not problems:
            await self._passthrough(payload, repair_out, run_dir, stage_dir, started, "全部达标", details)
            return payload

        grid = self.load_json("inpaint_grid.json")
        overrides = grid.get("region_overrides") or {}
        base_plan = dict(mask.get("plan") or payload.get("repair_plan") or {})

        # 每个问题区域带自己的 mask 覆盖参数
        detect = payload.get("detect") or {}
        work_candidates = [
            {
                "id": r["id"],
                "bbox": r["bbox_work"],
                "plan_override": {
                    k: v
                    for k, v in (overrides.get(problems.get(r["id"], ""), {}) or {}).items()
                    if k in {"ink_threshold", "dilate_extra", "pad_ratio", "feather_sigma"}
                },
            }
            for r in (detect.get("regions") or [])
            if r.get("bbox_work")
        ]
        if not work_candidates:  # 兜底：用 mask 里已有的全分辨率框
            self.completed(payload, "缺少工作坐标，跳过精修", applied=False)
            payload["refine"] = {"applied": False, "output": str(repair_out), "details": details}
            return payload

        meta = detect.get("video") or {}
        orig_size = (int(meta.get("width") or 0), int(meta.get("height") or 0))
        scale = float((payload.get("_sample_meta") or {}).get("scale") or 1.0)
        maps = payload.get("_maps")
        if maps is None or not orig_size[0]:
            self.completed(payload, "缺少 support 图/视频尺寸，跳过精修", applied=False)
            payload["refine"] = {"applied": False, "output": str(repair_out), "details": details}
            return payload

        refined_masks = await self.to_thread(
            wu.build_masks, maps, work_candidates, orig_size, scale, base_plan
        )

        # 全局参数取问题最严重的那一类（inpaint 的 radius 是全局的）
        params = dict(repair.get("params") or {})
        if "edge_residual" in problems.values():
            override = overrides.get("edge_residual", {})
            params.update(
                {
                    "method": override.get("method", "ns"),
                    "radius": int(override.get("radius", 8)),
                    "blur_mix": float(override.get("blur_mix", 0.35)),
                }
            )
        elif "over_smooth" in problems.values():
            override = overrides.get("over_smooth", {})
            params.update(
                {
                    "method": override.get("method", "telea"),
                    "radius": int(override.get("radius", 4)),
                    "blur_mix": float(override.get("blur_mix", 0.0)),
                }
            )
        # 是否启用增量（patch 移植 / 纹理回填）只看当前轮次的 plan：
        # 首轮是 baseline，精修也用 baseline，质检不过才由回灌轮打开
        params["patch_fill"] = bool(base_plan.get("patch_fill", True))
        params["texture"] = bool(base_plan.get("texture", True))
        params["texture_gain"] = float(base_plan.get("texture_gain", 3.0))

        out_path = run_dir / "clean.mp4"
        if out_path.exists():
            out_path.unlink()
        last_percent = -100.0

        def on_progress(done: int, total: int, message: str) -> None:
            nonlocal last_percent
            percent = 100.0 * done / max(total, 1)
            if percent - last_percent < 3.0 and done != total:
                return
            last_percent = percent
            self.progress(payload, f"精修重修：{message}", percent=round(percent, 1))

        stats2 = await self.to_thread(
            wu.inpaint_video,
            src,
            out_path,
            refined_masks,
            {**params, "crf": max(16, config.VIDEO_CRF - 2)},
            on_progress,
            6,
        )
        shutil.copyfile(out_path, stage_dir / "clean.mp4")

        payload["refine"] = {
            "applied": True,
            "output": str(out_path),
            "params": params,
            "problems": problems,
            "details": details,
            "stats": stats2,
            "elapsed": round(time.perf_counter() - started, 3),
        }
        payload["trace"] = payload.get("trace", [])
        payload["trace"].append(
            {
                "agent": self.name,
                "seconds": round(time.perf_counter() - started, 3),
                "applied": True,
                "problems": len(problems),
            }
        )
        self.completed(
            payload,
            f"精修完成：{len(problems)} 个区域重修（{params.get('method')}/r{params.get('radius')}）",
            applied=True,
            problems=len(problems),
        )
        return payload

    # ------------------------------------------------------------------ #
    async def _passthrough(
        self,
        payload: dict[str, Any],
        repair_out: Path,
        run_dir: Path,
        stage_dir: Path,
        started: float,
        reason: str,
        details: list[dict[str, Any]] | None = None,
    ) -> None:
        """无需精修：把修复结果直接作为成品（不重编码，不二次压缩）。"""
        out_path = run_dir / "clean.mp4"
        if repair_out.exists() and repair_out.resolve() != out_path.resolve():
            shutil.copyfile(repair_out, out_path)
        target = stage_dir / "clean.mp4"
        if out_path.exists() and target.resolve() != out_path.resolve():
            shutil.copyfile(out_path, target)
        payload["refine"] = {
            "applied": False,
            "output": str(out_path),
            "reason": reason,
            "details": details or [],
            "elapsed": round(time.perf_counter() - started, 3),
        }
        payload["trace"] = payload.get("trace", [])
        payload["trace"].append(
            {"agent": self.name, "seconds": round(time.perf_counter() - started, 3), "applied": False}
        )
        self.completed(payload, f"无需精修（{reason}），成品直接沿用修复结果", applied=False)