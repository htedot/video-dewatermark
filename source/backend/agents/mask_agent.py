"""遮罩 Agent：把检测框变成「实心修复 mask + 羽化 alpha」。

* 实心 mask 给 ``cv2.inpaint`` 用（二值，0/255）；
* 羽化 mask 用 ``cv2.GaussianBlur(binary, (0,0), sigma=5~10)`` 得到，回贴时做软合成，
  这样修复区与四周不会出现硬边框（规划里的"羽化 mask 高斯 5-10px"）；
* 参数来自 ``backend/data/inpaint_grid.json`` 的梯度表；回灌时用 reviewer 给的
  ``retry_plan`` 覆盖（加厚 mask / 降阈值等）。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import config
import watermark_utils as wu
from agents.base import BaseAgent


class MaskAgent(BaseAgent):
    name = "mask"
    label = "遮罩 Agent"
    description = "把检测框扩成羽化 mask（高斯 5-10px），保证修复边界无硬缝"

    async def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        run_dir = self.run_dir(payload)
        detect = payload.get("detect") or {}
        regions = detect.get("regions") or []
        attempt = int(payload.get("attempt", 1))
        self.started(payload, f"生成羽化遮罩（第 {attempt} 轮）")

        grid = self.load_json("inpaint_grid.json")
        plan = self._resolve_plan(payload, grid, attempt)
        payload["repair_plan"] = plan

        meta = detect.get("video") or {}
        orig_size = (int(meta.get("width") or 0), int(meta.get("height") or 0))
        sample_meta = payload.get("_sample_meta") or {}
        scale = float(sample_meta.get("scale") or 1.0)

        if not regions or not orig_size[0]:
            empty = {
                "binary": None,
                "feather": None,
                "regions": [],
                "coverage": 0.0,
                "plan": plan,
            }
            payload["mask"] = empty
            payload["trace"] = payload.get("trace", [])
            payload["trace"].append({"agent": self.name, "seconds": round(time.perf_counter() - started, 3),
                                     "regions": 0})
            self.completed(payload, "没有需要修复的区域，跳过遮罩生成", regions=0)
            return payload

        maps = payload.get("_maps")
        if maps is None:
            raise RuntimeError("缺少 _maps：mask 环节依赖检测环节的 support 图")

        work_candidates = [
            {
                "id": r["id"],
                "bbox": r["bbox_work"],
                # 回灌时 reviewer 会针对问题区域给 mask 覆盖参数（加厚/收缩）
                "plan_override": {
                    k: v
                    for k, v in ((payload.get("retry_overrides") or {}).get(r["id"]) or {}).items()
                    if k in {"ink_threshold", "dilate_extra", "pad_ratio", "feather_sigma"}
                },
            }
            for r in regions
            if r.get("bbox_work")
        ]
        masks = await self.to_thread(
            wu.build_masks, maps, work_candidates, orig_size, scale, plan
        )

        # 落盘：二值 mask + 羽化热力图 + 叠加图，便于人工核对
        artifacts: list[str] = []
        try:
            preview = wu.mask_preview(
                masks["binary"], masks["feather"]
            )
            mask_path = run_dir / "mask.png"
            await self.to_thread(wu.write_image, mask_path, masks["binary"])
            preview_path = run_dir / "mask_heat.png"
            await self.to_thread(wu.write_image, preview_path, preview)
            artifacts.extend([str(mask_path), str(preview_path)])
        except Exception as exc:
            self.progress(payload, f"mask 配图失败（不影响主流程）：{exc}", percent=80)

        payload["mask"] = {
            "binary": masks["binary"],
            "feather": masks["feather"],
            "regions": masks["regions"],
            "coverage": masks["coverage"],
            "plan": plan,
            "artifacts": artifacts,
        }
        payload["trace"] = payload.get("trace", [])
        payload["trace"].append(
            {
                "agent": self.name,
                "seconds": round(time.perf_counter() - started, 3),
                "regions": len(masks["regions"]),
                "coverage": masks["coverage"],
            }
        )
        self.completed(
            payload,
            f"遮罩就绪：{len(masks['regions'])} 个区域，覆盖画面 {masks['coverage'] * 100:.2f}%，"
            f"羽化 σ={plan.get('feather_sigma')}",
            regions=len(masks["regions"]),
            coverage=masks["coverage"],
        )
        return payload

    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_plan(payload: dict[str, Any], grid: dict[str, Any], attempt: int) -> dict[str, Any]:
        """默认取梯度表第 attempt 档；回灌时用 reviewer 的 retry_plan 覆盖。"""
        steps = grid.get("grid") or []
        base = dict(steps[min(max(attempt - 1, 0), max(len(steps) - 1, 0))]) if steps else {}
        base.pop("why", None)
        retry = payload.get("retry_plan") or {}
        for key in ("method", "radius", "blur_mix", "feather_sigma", "dilate_extra",
                    "ink_threshold", "pad_ratio", "attenuation_ratio", "texture_gain",
                    "patch_fill", "texture"):
            if key in retry and retry[key] is not None:
                base[key] = retry[key]
        # 兜底默认值（梯度表缺失时）
        base.setdefault("feather_sigma", config.MASK_FEATHER_SIGMA)
        base.setdefault("dilate_extra", 0)
        base.setdefault("ink_threshold", 0.55)
        base.setdefault("pad_ratio", 0.10)
        base.setdefault("attenuation_ratio", config.MASK_ATTENUATION_RATIO)
        base.setdefault("method", "telea")
        base.setdefault("radius", config.INPAINT_RADIUS)
        base.setdefault("blur_mix", 0.0)
        base.setdefault("patch_fill", True)
        base.setdefault("texture", True)
        base.setdefault("texture_gain", 3.0)
        base["attempt"] = attempt
        return base