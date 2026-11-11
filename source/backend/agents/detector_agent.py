"""检测 Agent：多帧投票找静态水印/台标，输出 bbox + 置信度。

流程：
1. 抽样 N 帧灰度图（默认 48 帧，工作宽度 960）；
2. 逐帧 Canny → 累加"静态边缘支持率图" support；
3. support >= 0.7 的像素连成候选块 → 面积/边长/填充比过滤 → 置信度；
4. 规则先给一份取舍（min_confidence + top1 保底），DeepSeek 只在候选内做复核；
5. 坐标换算回原分辨率，写进 payload["detect"]。

LLM 只做"取舍与归类"，不允许新增 id / 改动数字；返回的 id 必须在上游候选里，
否则整份结果作废回退规则（这就是规划里说的"质检做原文回溯"）。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import config
import watermark_utils as wu
from agents.base import BaseAgent


class DetectorAgent(BaseAgent):
    name = "detector"
    label = "检测 Agent"
    description = "多帧投票定位静态水印/台标（出现率 > 70%），输出 bbox 与置信度"

    # ------------------------------------------------------------------ #
    async def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        run_dir = self.run_dir(payload)
        src = Path(payload["src_path"])
        self.started(payload, "抽样帧，做多帧静态边缘投票")

        # 可选：本次请求的参数覆盖（前端滑块），不传就用 config 默认值
        req = payload.get("params") or {}
        sample_frames = max(8, min(120, int(req.get("sample_frames", config.DETECT_SAMPLE_FRAMES))))
        appear_rate = float(req.get("appear_rate", config.DETECT_APPEAR_RATE))
        appear_rate = max(0.30, min(0.98, appear_rate))

        sample = await self.to_thread(
            wu.read_sample_gray, src, sample_frames, config.DETECT_WORK_WIDTH
        )
        meta = sample["meta"]
        self.progress(
            payload,
            f"抽到 {sample['count']} 帧（{meta['width']}x{meta['height']} @ {meta['fps']}fps）"
            f"，工作分辨率 {sample['work_size'][0]}x{sample['work_size'][1]}",
            percent=15,
            sample_count=sample["count"],
        )

        maps = await self.to_thread(wu.compute_static_maps, sample["grays"])
        presets = self.load_json("watermark_presets.json")
        candidates = await self.to_thread(
            wu.generate_candidates, maps, presets, appear_rate
        )
        self.progress(
            payload,
            f"多帧投票得到 {len(candidates)} 个候选（出现率阈值 {appear_rate:.0%}）",
            percent=45,
            candidates=len(candidates),
        )

        # ---------------- 规则先挑，LLM 再复核 ---------------- #
        policy = dict(presets.get("select_policy", {}))
        if req.get("max_regions") is not None:
            policy["max_regions"] = max(1, min(10, int(req["max_regions"])))
        if req.get("min_conf") is not None:
            policy["min_confidence"] = max(0.0, min(0.95, float(req["min_conf"])))
        min_conf = float(policy.get("min_confidence", config.DETECT_MIN_CONF))
        rule_selected, rule_kinds = self._rule_select(candidates, min_conf, policy)
        selected_ids = list(rule_selected)
        decisions: dict[str, dict[str, Any]] = dict(rule_kinds)
        review_source = "rules"

        if candidates:
            llm_ids, llm_notes = await self._llm_review(
                payload, meta, candidates, presets, sample_frames
            )
            if llm_ids is not None:
                selected_ids = llm_ids
                for cid in llm_ids:
                    if cid in llm_notes:
                        decisions[cid] = llm_notes[cid]
                review_source = "llm"
                self.progress(
                    payload,
                    f"LLM 复核：保留 {len(selected_ids)} 个区域",
                    percent=70,
                    selected=len(selected_ids),
                )

        # 保底：规则选中的 top1 不允许被 LLM 全部否掉（否则"什么都没修"更难解释）
        if rule_selected and not selected_ids:
            selected_ids = rule_selected[:1]
            review_source += "+fallback_top1"
            self.progress(payload, "LLM 全部否决，按策略保底保留置信度最高的 1 个区域", percent=72)

        scale = float(sample["scale"])
        selected_map = {c["id"]: c for c in candidates if c["id"] in set(selected_ids)}
        regions: list[dict[str, Any]] = []
        for cid in selected_ids:
            cand = selected_map.get(cid)
            if cand is None:
                continue
            x, y, w, h = [float(v) for v in cand["bbox"]]
            regions.append(
                {
                    "id": cid,
                    "bbox": [
                        int(round(x / scale)),
                        int(round(y / scale)),
                        int(round(w / scale)),
                        int(round(h / scale)),
                    ],
                    "bbox_work": [int(round(v)) for v in cand["bbox"]],
                    "kind": decisions.get(cid, {}).get("kind", "unknown"),
                    "confidence": cand["confidence"],
                    "appear_rate": cand["appear_rate"],
                    "zone": cand["zone"],
                    "reason": decisions.get(cid, {}).get("reason", ""),
                }
            )
        regions.sort(key=lambda r: (-r["confidence"], r["bbox"][1], r["bbox"][0]))
        for i, region in enumerate(regions, start=1):
            region["index"] = i

        # ---------------- 报告配图（自证：投票热力图 + 框选图） ---------------- #
        artifacts: list[str] = []
        try:
            heat = wu.mask_preview(
                (maps["support"] >= appear_rate).astype("uint8") * 255,
                maps["support"],
            )
            heat_path = run_dir / "detect_support.jpg"
            await self.to_thread(wu.write_image, heat_path, heat)
            artifacts.append(str(heat_path))

            mid = sample["indices"][len(sample["indices"]) // 2]
            frames = await self.to_thread(wu.read_frames_at, src, [mid])
            if frames:
                # 候选框工作坐标 → 原图：annotate_boxes 内部按 1/scale 放大
                annotated = wu.annotate_boxes(frames[0], candidates, scale)
                annot_path = run_dir / "detect_preview.jpg"
                await self.to_thread(wu.write_image, annot_path, annotated)
                artifacts.append(str(annot_path))
        except Exception as exc:  # 配图失败不影响主流程
            self.progress(payload, f"检测配图生成失败（不影响主流程）：{exc}", percent=80)

        detect = {
            "video": meta,
            "sample": {
                "frames": sample["count"],
                "indices": sample["indices"],
                "work_size": list(sample["work_size"]),
                "scale": round(scale, 6),
                "appear_rate_threshold": round(appear_rate, 4),
            },
            "candidates": candidates,
            "selected": [r["id"] for r in regions],
            "regions": regions,
            "review_source": review_source,
            "policy": policy,
            "artifacts": artifacts,
            "elapsed": round(time.perf_counter() - started, 3),
        }
        payload["detect"] = detect
        payload["_maps"] = maps           # mask 环节直接复用，避免重复解码整段视频
        payload["_sample_meta"] = {
            "scale": scale,
            "work_size": list(sample["work_size"]),
            "indices": sample["indices"],
        }
        payload["trace"] = payload.get("trace", [])
        payload["trace"].append(
            {"agent": self.name, "seconds": round(time.perf_counter() - started, 3),
             "regions": len(regions), "candidates": len(candidates)}
        )

        if regions:
            summary = "、".join(
                f"{r['id']}@{r['zone']}({r['confidence']:.2f})" for r in regions[:4]
            )
            message = f"检测到 {len(regions)} 个静态水印区域：{summary}"
        else:
            message = f"未检测到稳定的静态水印（画面里没有出现率 >{appear_rate:.0%} 的固定边缘）"
        self.completed(payload, message, regions=len(regions), candidates=len(candidates))
        return payload

    # ------------------------------------------------------------------ #
    def _rule_select(
        self, candidates: list[dict[str, Any]], min_conf: float, policy: dict
    ) -> tuple[list[str], dict[str, dict[str, Any]]]:
        """规则取舍：置信度过滤 + 数量上限 + top1 保底。"""
        max_regions = int(policy.get("max_regions", 6))
        keep_top = int(policy.get("always_keep_top", 1))
        ordered = sorted(candidates, key=lambda c: -c["confidence"])
        picked: list[str] = []
        notes: dict[str, dict[str, Any]] = {}
        for cand in ordered:
            if len(picked) >= max_regions:
                break
            if cand["confidence"] >= min_conf or len(picked) < keep_top:
                picked.append(cand["id"])
                notes[cand["id"]] = {
                    "kind": self._guess_kind(cand),
                    "reason": f"规则：置信度 {cand['confidence']:.2f}，位于 {cand['zone']}",
                }
        return picked, notes

    @staticmethod
    def _guess_kind(cand: dict[str, Any]) -> str:
        w, h = float(cand["bbox"][2]), float(cand["bbox"][3])
        ratio = w / max(h, 1.0)
        if cand["zone"] in {"top_corner", "bottom_corner"} and ratio < 4.5:
            return "logo"
        if ratio >= 4.5:
            return "text"
        if cand["area_ratio"] < 0.002:
            return "badge"
        return "unknown"

    # ------------------------------------------------------------------ #
    async def _llm_review(
        self,
        payload: dict[str, Any],
        meta: dict[str, Any],
        candidates: list[dict[str, Any]],
        presets: dict[str, Any],
        sample_frames: int,
    ) -> tuple[list[str] | None, dict[str, dict[str, Any]]]:
        """让 DeepSeek 在候选内做取舍。返回 (selected_ids|None, id→{kind,reason})。"""
        system = self.read_prompt("detector_system.txt")
        if not system:
            return None, {}
        # 只把结构化、已裁剪的字段给它；数字保留两位小数，避免它复述长浮点
        view = [
            {
                "id": c["id"],
                "bbox_norm": c["bbox_norm"],
                "appear_rate": c["appear_rate"],
                "stability": c["stability"],
                "edge_strength": c["edge_strength"],
                "contrast": c["contrast"],
                "zone": c["zone"],
                "confidence": c["confidence"],
                "area_ratio": c["area_ratio"],
                "fill_ratio": c["fill_ratio"],
                "inner_tstd": c["inner_tstd"],
            }
            for c in candidates
        ]
        user = (
            "下面是一次真实视频检测的输出。请判断每个候选是不是「静态叠加的水印/台标」，"
            "并给出 kind（logo/text/badge/unknown）与一句理由。\n"
            "只允许引用已有 id；顶层输出 {\"selected\":[{\"id\",\"kind\",\"reason\"}],"
            "\"rejected\":[{\"id\",\"reason\"}]}，不要输出 JSON 之外的任何文字。\n\n"
            f"视频信息：{meta['width']}x{meta['height']}，{meta['duration']} 秒，"
            f"{meta['fps']}fps，抽样 {sample_frames} 帧。\n"
            f"规则库 select_policy：{presets.get('select_policy', {})}\n"
            f"位置先验 zones：{presets.get('zones', {})}\n"
            f"候选列表（JSON）：{view}"
        )

        def _parse(data: Any) -> dict[str, Any]:
            if not isinstance(data, dict):
                raise ValueError("LLM 顶层不是 dict")
            return data

        result, _tokens = await self.call_llm(
            payload, "detector_review", system, user, parse=_parse, json_mode=True
        )
        if not isinstance(result, dict):
            return None, {}

        valid = {c["id"] for c in candidates}
        kinds_allowed = set((presets.get("kinds") or {}).keys()) | {"unknown"}
        notes: dict[str, dict[str, Any]] = {}
        selected: list[str] = []
        for item in result.get("selected") or []:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("id", "")).strip()
            if cid not in valid:
                continue  # 编造的 id 直接丢（原文回溯）
            kind = str(item.get("kind", "unknown"))
            if kind not in kinds_allowed:
                kind = "unknown"
            notes[cid] = {"kind": kind, "reason": str(item.get("reason", ""))[:120]}
            selected.append(cid)
        for item in result.get("rejected") or []:
            if isinstance(item, dict):
                cid = str(item.get("id", "")).strip()
                if cid in valid and cid not in notes:
                    notes[cid] = {"kind": "unknown", "reason": "LLM: " + str(item.get("reason", ""))[:100]}
        # 去重保序
        deduped = list(dict.fromkeys(selected))
        return deduped, notes