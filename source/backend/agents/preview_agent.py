"""预览 Agent（并行节点）：把「原片 vs 成品」拼成一张对比图，给人快速判断。

和报告 Agent 互不依赖，两个一起在并行的 reporting 层里跑（asyncio.gather 聚合），
一个写图、一个写 JSON，互不抢同一个文件。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import watermark_utils as wu
from agents.base import BaseAgent


class PreviewAgent(BaseAgent):
    name = "preview"
    label = "对比预览 Agent"
    description = "生成原片/成品的对比图，便于人工快速验收"

    async def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        run_dir = self.run_dir(payload)
        self.started(payload, "生成原片/成品对比预览")

        refine = payload.get("refine") or {}
        final_path = Path(refine.get("output") or run_dir / "clean.mp4")
        src_path = Path(payload["src_path"])
        detect = payload.get("detect") or {}
        regions = detect.get("regions") or []
        scale = float((payload.get("_sample_meta") or {}).get("scale") or 1.0)

        artifacts: list[str] = []
        panels: list[np.ndarray] = []
        labels: list[str] = []
        try:
            info = await self.to_thread(wu.probe_video, final_path)
            indices = wu.evenly_spaced_indices(int(info.get("frame_count") or 0), 3)
            before = await self.to_thread(wu.read_frames_at, src_path, indices)
            after = await self.to_thread(wu.read_frames_at, final_path, indices)
            for i, (b, a) in enumerate(zip(before, after), start=1):
                if regions:
                    b = wu.annotate_boxes(b, regions, scale)
                # 统一到同一尺寸再拼（编码可能改了分辨率）
                if b.shape[:2] != a.shape[:2]:
                    a = cv2.resize(a, (b.shape[1], b.shape[0]))
                panels.extend([b, a])
                labels.extend([f"原片 #{i}（检测框）", f"成品 #{i}"])
            if panels:
                montage = wu.side_by_side(panels, labels, tile_height=260)
                header = np.full((30, montage.shape[1], 3), 18, np.uint8)
                cv2.putText(
                    header,
                    f"原片 vs 成品：{len(regions)} 个区域，抽样帧 {indices}",
                    (10, 21),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )
                montage = np.vstack([header, montage])
                path = run_dir / "preview_before_after.jpg"
                await self.to_thread(wu.write_image, path, montage)
                artifacts.append(str(path))
        except Exception as exc:  # 预览失败不影响任务成败
            self.progress(payload, f"对比预览生成失败（不影响任务）：{exc}", percent=50)

        payload["preview"] = {
            "artifacts": artifacts,
            "elapsed": round(time.perf_counter() - started, 3),
        }
        payload["trace"] = payload.get("trace", [])
        payload["trace"].append(
            {"agent": self.name, "seconds": round(time.perf_counter() - started, 3),
             "artifacts": len(artifacts)}
        )
        self.completed(payload, "对比预览已生成", artifacts=len(artifacts))
        return payload