"""修复 Agent：cv2.inpaint（TELEA / NS / 局部模糊兜底）逐帧修复并重新编码。

* 单次解码 → 逐帧修 → rawvideo 管道喂 ffmpeg 编码 H.264，音频直接 ``-c:a copy``；
* 参数来自遮罩环节的 ``repair_plan``，可用 DeepSeek 在**允许集合内**调整
  （method / radius / blur_mix），越界即回退默认值；
* 每 6 帧抽一帧算一次区域质量指标，写进 ``payload["repair"]["stats"]``，
  供 Refiner 判断"哪个区域还需要重修"。
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import config
import watermark_utils as wu
from agents.base import BaseAgent


class InpainterAgent(BaseAgent):
    name = "inpainter"
    label = "修复 Agent"
    description = "逐帧 cv2.inpaint 修复水印区域，音频无损保留"

    async def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        run_dir = self.run_dir(payload)
        stage_dir = Path(payload.get("stage_dir") or run_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)
        src = Path(payload["src_path"])
        attempt = int(payload.get("attempt", 1))
        self.started(payload, f"开始逐帧修复（第 {attempt} 轮）")

        grid = self.load_json("inpaint_grid.json")
        plan = dict(payload.get("repair_plan") or {})
        params = await self._choose_params(payload, plan, grid)
        mask = payload.get("mask") or {}
        binary = mask.get("binary")
        has_regions = binary is not None and getattr(binary, "any", lambda: False)()

        out_path = stage_dir / "inpainted.mp4"
        if not has_regions:
            # 没有区域：直接复用原片（不重编码，画质不损失）
            if out_path.exists():
                out_path.unlink()
            await self.to_thread(shutil.copyfile, src, out_path)
            repair = {
                "params": params,
                "output": str(out_path),
                "stats": {"frames_written": 0, "skipped": True, "regions": []},
                "note": "没有检测到区域，成品即原片副本",
                "elapsed": round(time.perf_counter() - started, 3),
            }
            payload["repair"] = repair
            payload["trace"] = payload.get("trace", [])
            payload["trace"].append({"agent": self.name, "seconds": repair["elapsed"], "skipped": True})
            self.completed(payload, "无区域可修，跳过修复", skipped=True)
            return payload

        last_percent = -100.0

        def on_progress(done: int, total: int, message: str) -> None:
            nonlocal last_percent
            percent = 100.0 * done / max(total, 1)
            if percent - last_percent < 2.0 and done != total:
                return
            last_percent = percent
            self.progress(payload, message, percent=round(percent, 1), frames=done, total=total)

        crf = config.VIDEO_CRF if not payload.get("refining") else max(16, config.VIDEO_CRF - 4)
        stats = await self.to_thread(
            wu.inpaint_video, src, out_path, mask, {**params, "crf": crf}, on_progress, 6
        )

        repair = {
            "params": params,
            "output": str(out_path),
            "stats": stats,
            "elapsed": round(time.perf_counter() - started, 3),
        }
        payload["repair"] = repair
        payload["trace"] = payload.get("trace", [])
        payload["trace"].append(
            {
                "agent": self.name,
                "seconds": repair["elapsed"],
                "frames": stats.get("frames_written"),
                "method": params.get("method"),
                "radius": params.get("radius"),
            }
        )
        self.completed(
            payload,
            f"修复完成：{stats.get('frames_written', 0)} 帧，参数 {params.get('method')}/r{params.get('radius')}"
            f"，音频{'保留' if (payload.get('detect') or {}).get('video', {}).get('has_audio') else '无'}",
            frames=stats.get("frames_written"),
            method=params.get("method"),
            radius=params.get("radius"),
        )
        return payload

    # ------------------------------------------------------------------ #
    async def _choose_params(
        self, payload: dict[str, Any], plan: dict[str, Any], grid: dict[str, Any]
    ) -> dict[str, Any]:
        """规则给默认参数；有 Key 时让 LLM 在允许集合内挑一组。"""
        params = {
            "method": str(plan.get("method", "telea")),
            "radius": int(plan.get("radius", config.INPAINT_RADIUS)),
            "blur_mix": float(plan.get("blur_mix", 0.0)),
            # 本项目增量：patch 移植 + 纹理回填。默认只在质检回灌的重修轮启用
            # （首轮走任务书 baseline 的 cv2.inpaint，便宜；不达标再付增量的成本）
            "patch_fill": bool(plan.get("patch_fill", True)),
            "texture": bool(plan.get("texture", True)),
            "texture_sigma": float(plan.get("texture_sigma", 8.0)),
            "texture_gain": float(plan.get("texture_gain", 3.0)),
        }
        system = self.read_prompt("inpainter_system.txt")
        if not system:
            return params
        regions = ((payload.get("detect") or {}).get("regions") or [])
        view = [
            {
                "id": r["id"],
                "zone": r["zone"],
                "confidence": r["confidence"],
                "kind": r.get("kind"),
                "size_px": r["bbox"][2:],
            }
            for r in regions
        ]
        feedback = payload.get("last_failure") or "首次尝试，无历史指标"
        user = (
            "请在允许集合内选一组修复参数。\n"
            "允许集合："
            f"method={grid.get('allowed_methods')}，radius={grid.get('allowed_radius')}，"
            f"blur_mix={grid.get('allowed_blur_mix')}。\n"
            f"默认建议：{params}\n"
            f"待修复区域（JSON）：{view}\n"
            f"上一轮质检反馈：{feedback}\n"
            '输出格式：{"params":{"method":"telea","radius":5,"blur_mix":0.0},"reason":"一句中文"}\n'
            "只输出 JSON 对象，不要 Markdown 围栏或额外文字。"
        )

        def _parse(data: Any) -> dict[str, Any]:
            if not isinstance(data, dict):
                raise ValueError("顶层不是 dict")
            return data

        result, _tokens = await self.call_llm(
            payload, "inpainter_params", system, user, parse=_parse, json_mode=True
        )
        if not isinstance(result, dict):
            return params
        picked = result.get("params") if isinstance(result.get("params"), dict) else result
        chosen = {
            "method": str(picked.get("method", params["method"])).lower(),
            "radius": int(picked.get("radius", params["radius"]) or params["radius"]),
            "blur_mix": float(picked.get("blur_mix", params["blur_mix"]) or 0.0),
            "source": "llm",
        }
        # 严格校验：只要越界就回退默认值（不允许 LLM 输出危险参数）
        if chosen["method"] not in (grid.get("allowed_methods") or ["telea"]):
            chosen["method"] = params["method"]
        allowed_r = grid.get("allowed_radius") or [3, 4, 5, 6, 7, 8]
        if chosen["radius"] not in allowed_r:
            chosen["radius"] = min(allowed_r, key=lambda v: abs(v - chosen["radius"]))
        allowed_m = grid.get("allowed_blur_mix") or [0.0]
        if chosen["blur_mix"] not in allowed_m:
            chosen["blur_mix"] = min(allowed_m, key=lambda v: abs(v - chosen["blur_mix"]))
        params = {k: chosen[k] for k in ("method", "radius", "blur_mix")}
        params.update(
            {
                "patch_fill": bool(plan.get("patch_fill", True)),
                "texture": bool(plan.get("texture", True)),
                "texture_sigma": float(plan.get("texture_sigma", 8.0)),
                "texture_gain": float(plan.get("texture_gain", 3.0)),
            }
        )
        params["source"] = "llm"
        self.progress(
            payload,
            f"LLM 选参：{params['method']} radius={params['radius']} blur_mix={params['blur_mix']}",
            percent=5,
        )
        return params