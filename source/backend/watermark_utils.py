"""水印检测 / 遮罩 / 修复 / 质量度量的纯函数集合（不含 Agent 逻辑，便于单测）。

算法主线（对应规划文档 6、7 节）：

1. **多帧投票检测**
   抽样 N 帧 → 逐帧 Canny 边缘 → 累加成"静态边缘支持率图" ``support``。
   真实画面在动，边缘位置每帧都在变；台标/静态水印的边缘每帧都落在同一处，
   于是 ``support >= 0.7`` 的像素就是"出现率 > 70% 的边缘"，即水印候选。
   再用形态学闭运算把笔画连成字/块，连通域 → bbox，多帧出现率 + 边缘强度 +
   时间稳定性 + 位置先验 → 置信度；重叠框合并。

2. **羽化 mask**
   bbox 内取 ``support`` 高值笔画 → 闭运算 + 填洞 → 得到实心字形 mask；
   整体再做 ``cv2.GaussianBlur(mask, (0,0), sigma=5~10)`` 得到羽化 alpha，
   修复时用二值 mask 进 ``cv2.inpaint``，回贴时用羽化 alpha 做软合成（边缘无硬缝）。

3. **修复**
   ``cv2.inpaint``（TELEA / NS，radius 3-8）；纹理过大时按 ``blur_mix`` 与局部模糊
   混合兜底（大块区域纯 inpaint 容易糊出不自然纹理）。

4. **质量度量**
   逐帧对比"修复区方差 vs 邻域环带方差"（``|Δvar|/var_ring < 0.3``）与
   "修复区边缘密度 vs 环带边缘密度"（残留水印边缘会让比值飙升），
   两者共同决定该区域是否合格。
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np

import config


class VideoError(RuntimeError):
    """视频打不开 / 编码失败。"""


# --------------------------------------------------------------------------- #
# 视频元信息
# --------------------------------------------------------------------------- #
def probe_video(path: str | Path) -> dict[str, Any]:
    """读视频元信息：优先 ffprobe（能拿到音轨/编码），失败退回 OpenCV。"""
    path = Path(path)
    info: dict[str, Any] = {
        "path": str(path),
        "width": 0,
        "height": 0,
        "fps": 25.0,
        "frame_count": 0,
        "duration": 0.0,
        "has_audio": False,
        "codec": "",
        "size_bytes": path.stat().st_size if path.is_file() else 0,
    }
    if config.FFPROBE:
        cmd = [
            config.FFPROBE,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=60)
            data = json.loads(proc.stdout.decode("utf-8", "replace") or "{}")
            streams = data.get("streams", [])
            video = next((s for s in streams if s.get("codec_type") == "video"), None)
            if video:
                info["width"] = int(video.get("width") or 0)
                info["height"] = int(video.get("height") or 0)
                info["codec"] = str(video.get("codec_name") or "")
                info["has_audio"] = any(s.get("codec_type") == "audio" for s in streams)
                rate = str(video.get("avg_frame_rate") or video.get("r_frame_rate") or "25/1")
                try:
                    num, _, den = rate.partition("/")
                    info["fps"] = float(num) / float(den or 1)
                except (ValueError, ZeroDivisionError):
                    info["fps"] = 25.0
                for key in ("nb_frames", "duration"):
                    raw = video.get(key) or data.get("format", {}).get(key)
                    if key == "nb_frames" and raw:
                        info["frame_count"] = int(raw)
                raw_dur = video.get("duration") or data.get("format", {}).get("duration")
                if raw_dur:
                    info["duration"] = float(raw_dur)
                info["fps"] = round(info["fps"], 3) or 25.0
                if not info["frame_count"] and info["duration"]:
                    info["frame_count"] = int(round(info["duration"] * info["fps"]))
                if info["width"] and info["height"]:
                    return info
        except (subprocess.SubprocessError, json.JSONDecodeError, ValueError, OSError):
            pass

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(f"无法打开视频：{path}")
    try:
        info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        info["fps"] = round(fps, 3) if 1.0 <= fps <= 240 else 25.0
        info["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if info["frame_count"] > 0 and info["fps"]:
            info["duration"] = round(info["frame_count"] / info["fps"], 2)
        if not info["width"] or not info["height"]:
            raise VideoError(f"读不到分辨率：{path}")
        return info
    finally:
        cap.release()


# --------------------------------------------------------------------------- #
# 抽帧 / 静态性分析
# --------------------------------------------------------------------------- #
def read_sample_gray(
    path: str | Path,
    count: int | None = None,
    work_width: int | None = None,
) -> dict[str, Any]:
    """抽样读灰度帧（检测用），返回帧列表 + 缩放系数 + 原始尺寸。"""
    count = int(count or config.DETECT_SAMPLE_FRAMES)
    work_width = int(work_width or config.DETECT_WORK_WIDTH)
    meta = probe_video(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(f"无法打开视频：{path}")
    try:
        total = meta["frame_count"]
        scale = min(1.0, work_width / float(meta["width"]))
        target = (max(16, int(round(meta["width"] * scale))),
                  max(16, int(round(meta["height"] * scale))))

        if total <= 0:
            # 帧数未知（部分容器）：顺序读，隔帧取
            frames, indices = [], []
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % 5 == 0:
                    frames.append(cv2.resize(frame, target, interpolation=cv2.INTER_AREA))
                    indices.append(idx)
                idx += 1
            if not frames:
                raise VideoError("视频里读不到任何帧")
        elif total <= count * 30:
            # 短视频：顺序解码最准（部分编码 seek 会偏帧）
            n = min(count, total)
            want = sorted({int(round(v)) for v in np.linspace(0, total - 1, n)})
            want_set = set(want)
            frames, indices = [], []
            for idx in range(total):
                ok, frame = cap.read()
                if not ok:
                    break
                if idx in want_set:
                    frames.append(cv2.resize(frame, target, interpolation=cv2.INTER_AREA))
                    indices.append(idx)
        else:
            # 长视频：跳帧 seek，够用且快
            n = min(count, total)
            want = sorted({int(round(v)) for v in np.linspace(0, total - 1, n)})
            frames, indices = [], []
            for idx in want:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
                ok, frame = cap.read()
                if not ok:
                    continue
                frames.append(cv2.resize(frame, target, interpolation=cv2.INTER_AREA))
                indices.append(idx)

        if not frames:
            raise VideoError("抽帧失败：视频可能是坏的或全是空帧")
        grays = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]).astype(np.uint8)
        return {
            "grays": grays,
            "indices": indices,
            "scale": scale,
            "work_size": target,
            "meta": meta,
            "count": int(grays.shape[0]),
        }
    finally:
        cap.release()


def compute_static_maps(grays: np.ndarray) -> dict[str, np.ndarray]:
    """静态边缘支持率图 + 时间标准差 + 中位帧。

    ``support``：每个像素在多少比例的抽样帧里被判为边缘（0~1），
    多帧投票的"出现率"就是它；水印区域会稳定接近 1，运动物体接近 0。
    """
    if grays.ndim == 3:
        stack = grays.astype(np.float32)
    else:  # 单帧兜底
        stack = grays[np.newaxis, ...].astype(np.float32)
    n = stack.shape[0]

    acc = np.zeros(stack.shape[1:], np.float32)
    grad_acc = np.zeros(stack.shape[1:], np.float32)
    for frame in stack:
        frame_u8 = frame.astype(np.uint8)
        smooth = cv2.GaussianBlur(frame_u8, (3, 3), 0)
        edges = cv2.Canny(smooth, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))  # 容忍 1px 抖动
        acc += (edges > 0).astype(np.float32)
        gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
        np.maximum(cv2.magnitude(gx, gy), grad_acc, out=grad_acc)

    support = acc / float(n)
    tstd = stack.std(axis=0)
    median = np.median(stack, axis=0).astype(np.uint8)
    return {
        "support": support,
        "tstd": tstd,
        "median": median,
        "grad": grad_acc / 4.0,  # Sobel 幅度归一化到 0~255 量级
        "frames": np.array([n], np.int32),
    }


def _local_contrast(median: np.ndarray) -> np.ndarray:
    """中位帧的局部对比度（拉普拉斯绝对值和），用来判断"这里有没有东西"。"""
    lap = cv2.Laplacian(cv2.GaussianBlur(median, (3, 3), 0), cv2.CV_32F, ksize=3)
    return cv2.GaussianBlur(np.abs(lap), (5, 5), 0)


def _box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _box_gap(a: Sequence[float], b: Sequence[float]) -> float:
    """两个框的最近距离（重叠为 0）。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = max(0.0, max(ax, bx) - min(ax + aw, bx + bw))
    dy = max(0.0, max(ay, by) - min(ay + ah, by + bh))
    return float(math.hypot(dx, dy))


def merge_boxes(
    boxes: list[list[float]],
    iou_thr: float = 0.05,
    gap_ratio: float = 0.02,
    frame_area: float | None = None,
) -> list[list[float]]:
    """把相邻/重叠的候选框合并（同一个台标被拆成多个连通域时）。"""
    items = [list(map(float, b)) for b in boxes]
    base = math.sqrt(frame_area) if frame_area else 100.0
    gap_thr = base * gap_ratio
    changed = True
    while changed and len(items) > 1:
        changed = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if _box_iou(items[i], items[j]) >= iou_thr or _box_gap(items[i], items[j]) <= gap_thr:
                    x1 = min(items[i][0], items[j][0])
                    y1 = min(items[i][1], items[j][1])
                    x2 = max(items[i][0] + items[i][2], items[j][0] + items[j][2])
                    y2 = max(items[i][1] + items[i][3], items[j][1] + items[j][3])
                    items[i] = [x1, y1, x2 - x1, y2 - y1]
                    items.pop(j)
                    changed = True
                    break
            if changed:
                break
    return items


def _position_prior(nx: float, ny: float, presets: dict) -> tuple[float, str]:
    """位置先验：台标 95% 在四角/边缘。入参是归一化中心坐标 (0~1)，返回 (0~1 权重, 区域名)。"""
    zones = presets.get("zones", {})
    margin = 0.22
    if ny <= 0.16 and (nx <= margin or nx >= 1 - margin):
        zone = "top_corner"
    elif ny >= 1 - 0.16 and (nx <= margin or nx >= 1 - margin):
        zone = "bottom_corner"
    elif ny <= 0.16:
        zone = "top_edge"
    elif ny >= 0.84:
        zone = "bottom_edge"
    elif nx <= 0.16 or nx >= 0.84:
        zone = "side_edge"
    else:
        zone = "center"
    weight = float(zones.get(zone, 0.5))
    return weight, zone


def _seed_ink(
    median: np.ndarray,
    support: np.ndarray,
    appear_rate: float,
    rules: dict[str, Any],
) -> tuple[np.ndarray, str]:
    """生成"候选种子"二值图：中位帧边缘 ∩ 多帧投票。

    只用 ``support >= 阈值`` 会把运动物体的残影也选进来（实测噪声连通域上百个），
    所以再加一道"中位帧上真的是条边"的约束：

    1. 主路径 ``median_edge+vote``：``Canny(中位帧) & (support >= appear_rate)``。
       水印文字/徽章在绝大多数帧里都是同一位置的强边，两个条件同时成立；
       运动内容在中位帧上是模糊残影，过不了 Canny 的阈值。
    2. 兜底 ``vote_only``：中位帧边缘为空（极低对比度水印）或主路径没选出任何连通域时，
       退回纯投票 ``support >= appear_rate``，至少不会漏检，只是噪声多些。
    """
    smooth = cv2.GaussianBlur(median, (3, 3), 0)
    lo = int(rules.get("seed_canny_low", 40))
    hi = int(rules.get("seed_canny_high", 120))
    edges = cv2.Canny(smooth, lo, hi)
    vote = (support >= appear_rate).astype(np.uint8) * 255
    ink = cv2.bitwise_and(edges, vote)
    if ink.any():
        return ink, "median_edge+vote"
    return vote, "vote_only"


def generate_candidates(
    maps: dict[str, np.ndarray],
    presets: dict | None = None,
    appear_rate: float | None = None,
) -> list[dict[str, Any]]:
    """静态边缘支持率图 → 水印候选框（工作分辨率坐标）+ 置信度。"""
    presets = presets or {}
    appear_rate = float(appear_rate if appear_rate is not None else config.DETECT_APPEAR_RATE)
    support = maps["support"]
    tstd = maps["tstd"]
    median = maps["median"]
    contrast = _local_contrast(median)
    height, width = support.shape
    frame_area = float(width * height)

    rules = presets.get("rules", {})
    kernel_close = int(rules.get("close_kernel", 9))
    kernel_dilate = int(rules.get("dilate_kernel", 5))
    min_area = float(rules.get("min_area_ratio", 0.00008)) * frame_area
    max_area = float(rules.get("max_area_ratio", 0.25)) * frame_area
    min_side = int(rules.get("min_side_px", 6))
    min_fill = float(rules.get("min_fill_ratio", 0.06))
    tstd_scale = max(1.0, float(rules.get("tstd_norm", 24.0)))
    grad_scale = max(1.0, float(rules.get("grad_norm", 60.0)))
    contrast_scale = max(1.0, float(rules.get("contrast_norm", 30.0)))
    weights = presets.get("weights", {})
    w_support = float(weights.get("appear_rate", 0.34))
    w_stable = float(weights.get("stability", 0.20))
    w_edge = float(weights.get("edge_strength", 0.16))
    w_pos = float(weights.get("position", 0.20))
    w_contrast = float(weights.get("contrast", 0.10))

    def _components_to_raw(seeds: np.ndarray, source: str) -> list[dict[str, Any]]:
        """种子二值图 → 连通域 → 候选（含置信度）。"""
        closed = cv2.morphologyEx(
            seeds, cv2.MORPH_CLOSE, np.ones((kernel_close, kernel_close), np.uint8)
        )
        closed = cv2.dilate(closed, np.ones((kernel_dilate, kernel_dilate), np.uint8))
        num, labels, stats, _centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
        found: list[dict[str, Any]] = []
        for idx in range(1, num):
            x, y, w, h, area = (int(v) for v in stats[idx][:5])
            if area < min_area or area > max_area:
                continue
            if w < min_side or h < min_side:
                continue
            if w / max(h, 1) > 25 or h / max(w, 1) > 25:
                continue
            fill = area / float(max(w * h, 1))
            if fill < min_fill:
                continue

            component = labels == idx
            # "出现率"只统计种子像素（真正的笔画边），否则会被框内大片背景稀释
            stroke = component & (seeds > 0)
            appear = float(support[stroke].mean()) if stroke.any() else float(support[component].mean())
            inner_tstd = float(tstd[y : y + h, x : x + w].mean())
            inner_contrast = float(contrast[y : y + h, x : x + w].mean())
            gradient = float(maps["grad"][y : y + h, x : x + w][component[y : y + h, x : x + w]].mean()) \
                if component.any() else 0.0

            stability = 1.0 - min(1.0, inner_tstd / tstd_scale)
            edge_norm = min(1.0, gradient / grad_scale)
            contrast_norm = min(1.0, inner_contrast / contrast_scale)
            prior, zone = _position_prior(
                (x + w / 2) / width, (y + h / 2) / height, presets
            )
            confidence = (
                w_support * min(1.0, appear)
                + w_stable * stability
                + w_edge * edge_norm
                + w_pos * prior
                + w_contrast * contrast_norm
            )
            found.append(
                {
                    "id": f"c{len(found) + 1}",
                    "bbox": [x, y, w, h],
                    "bbox_norm": [
                        round(x / width, 4),
                        round(y / height, 4),
                        round(w / width, 4),
                        round(h / height, 4),
                    ],
                    "area_ratio": round(area / frame_area, 6),
                    "appear_rate": round(min(1.0, appear), 4),
                    "stability": round(stability, 4),
                    "edge_strength": round(edge_norm, 4),
                    "contrast": round(contrast_norm, 4),
                    "zone": zone,
                    "position_prior": round(prior, 3),
                    "confidence": round(max(0.0, min(1.0, confidence)), 4),
                    "fill_ratio": round(fill, 3),
                    "inner_tstd": round(inner_tstd, 2),
                    "seed_source": source,
                }
            )
        return found

    ink, seed_source = _seed_ink(median, support, appear_rate, rules)
    raw = _components_to_raw(ink, seed_source) if ink.any() else []
    if not raw and seed_source != "vote_only":
        # 主路径一个连通域都没留下（低对比度水印 / 参数过严）→ 退回纯投票
        vote_ink = (support >= appear_rate).astype(np.uint8) * 255
        raw = _components_to_raw(vote_ink, "vote_only_fallback") if vote_ink.any() else []

    if not raw:
        return []

    merged_boxes = merge_boxes(
        [c["bbox"] for c in raw], frame_area=frame_area, gap_ratio=0.012
    )
    # 合并后重新算置信度：取被并入候选的加权平均
    candidates: list[dict[str, Any]] = []
    for box in merged_boxes:
        members = [
            c for c in raw
            if _box_iou(c["bbox"], box) > 0.05 or _box_gap(c["bbox"], box) < 1.0
        ]
        if not members:
            continue
        weight_sum = sum(max(m["confidence"], 0.05) for m in members)
        conf = sum(m["confidence"] * max(m["confidence"], 0.05) for m in members) / weight_sum
        x, y, w, h = box
        prior, zone = _position_prior(
            (x + w / 2) / width, (y + h / 2) / height, presets
        )
        candidates.append(
            {
                "id": "",
                "bbox": [int(round(v)) for v in box],
                "bbox_norm": [
                    round(x / width, 4),
                    round(y / height, 4),
                    round(w / width, 4),
                    round(h / height, 4),
                ],
                "area_ratio": round(float(w * h) / frame_area, 6),
                "appear_rate": round(float(np.mean([m["appear_rate"] for m in members])), 4),
                "stability": round(float(np.mean([m["stability"] for m in members])), 4),
                "edge_strength": round(float(np.mean([m["edge_strength"] for m in members])), 4),
                "contrast": round(float(np.mean([m["contrast"] for m in members])), 4),
                "zone": zone,
                "position_prior": round(prior, 3),
                "confidence": round(conf, 4),
                "fill_ratio": round(float(np.mean([m["fill_ratio"] for m in members])), 3),
                "inner_tstd": round(float(np.mean([m["inner_tstd"] for m in members])), 2),
                "merged_from": len(members),
                "seed_source": members[0].get("seed_source", ""),
            }
        )
    candidates.sort(key=lambda c: (c["bbox"][1], c["bbox"][0]))
    for i, item in enumerate(candidates, start=1):
        item["id"] = f"c{i}"
    return candidates


# --------------------------------------------------------------------------- #
# 遮罩
# --------------------------------------------------------------------------- #
def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """填掉连通域内部的洞（字形的空心部分）。"""
    h, w = mask.shape[:2]
    flood = mask.copy()
    pad = 1
    canvas = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    canvas[pad : pad + h, pad : pad + w] = flood
    ff = canvas.copy()
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 255)
    holes = cv2.bitwise_not(ff)[pad : pad + h, pad : pad + w]
    return cv2.bitwise_or(mask, holes)


def _overlay_attenuation(tstd: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    """估算「半透明叠加层」对框内时间方差的压低比例。

    不透明字幕只会占住笔画那几个像素，框内其余像素的时间方差与邻域持平；
    而半透明蒙版/徽章会把**整块**画面的时间方差按 (1-α)² 压低。
    所以 ``tstd_in_mean / tstd_ring_mean`` 明显小于 1，就说明整块区域都被覆盖了。
    """
    x, y, w, h = [int(v) for v in bbox]
    work_h, work_w = tstd.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(work_w, x + w), min(work_h, y + h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return 1.0
    px, py = max(3, int(w * 0.5)), max(3, int(h * 0.5))
    rx0, ry0 = max(0, x - px), max(0, y - py)
    rx1, ry1 = min(work_w, x + w + px), min(work_h, y + h + py)
    inner_mean = float(tstd[y0:y1, x0:x1].mean())
    ring_mean = float(tstd[ry0:ry1, rx0:rx1].mean())
    if ring_mean < 1e-3:
        return 1.0
    return inner_mean / ring_mean


def build_masks(
    maps: dict[str, np.ndarray],
    candidates: list[dict[str, Any]],
    orig_size: tuple[int, int],
    scale: float,
    plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """按候选框生成「实心修复 mask」+「羽化 alpha mask」（均在全分辨率坐标系）。

    返回 ``binary``（uint8 0/255，给 cv2.inpaint）、``feather``（float32 0~1，回贴用）、
    以及每个框的 mask 覆盖率，便于报告里核对"有没有把画面修坏"。
    """
    plan = plan or {}
    orig_w, orig_h = orig_size
    binary = np.zeros((orig_h, orig_w), np.uint8)
    support = maps["support"]
    work_h, work_w = support.shape
    sigma = float(plan.get("feather_sigma", config.MASK_FEATHER_SIGMA))
    ink_thr = float(plan.get("ink_threshold", 0.55))
    dilate_extra = int(plan.get("dilate_extra", 0))
    atten_ratio = float(plan.get("attenuation_ratio", config.MASK_ATTENUATION_RATIO))

    per_box: list[dict[str, Any]] = []
    for cand in candidates:
        # 回灌重修时，每个区域可以带自己的参数覆盖（加厚/收缩 mask）
        rplan = {**plan, **(cand.get("plan_override") or {})}
        ink_thr = float(rplan.get("ink_threshold", ink_thr))
        pad_ratio = float(rplan.get("pad_ratio", 0.10))
        dilate_extra = int(rplan.get("dilate_extra", dilate_extra))
        atten_ratio = float(rplan.get("attenuation_ratio", atten_ratio))
        sigma = max(sigma, float(rplan.get("feather_sigma", sigma)))
        x, y, w, h = [int(v) for v in cand["bbox"]]
        # 工作分辨率 → 全分辨率
        fx = int(round(x / scale))
        fy = int(round(y / scale))
        fw = int(round(w / scale))
        fh = int(round(h / scale))
        pad_x = max(3, int(round(fw * pad_ratio)))
        pad_y = max(3, int(round(fh * pad_ratio)))
        x0 = max(0, fx - pad_x)
        y0 = max(0, fy - pad_y)
        x1 = min(orig_w, fx + fw + pad_x)
        y1 = min(orig_h, fy + fh + pad_y)
        if x1 <= x0 or y1 <= y0:
            continue

        # 取工作分辨率下对应区域的笔画（support 高值）
        wx0 = max(0, int(math.floor(x0 * scale)))
        wy0 = max(0, int(math.floor(y0 * scale)))
        wx1 = min(work_w, int(math.ceil(x1 * scale)))
        wy1 = min(work_h, int(math.ceil(y1 * scale)))
        crop = support[wy0:wy1, wx0:wx1]
        if crop.size == 0:
            continue
        glyph = (crop >= ink_thr).astype(np.uint8) * 255
        if not glyph.any():
            # 退化：整框填满（宁多修不漏修）
            glyph = np.full(crop.shape, 255, np.uint8)

        # 半透明叠加层（蒙版/徽章/渐隐字幕）整块压低了时间方差：
        # 只盖笔画会留下"淡影 + 底色涂抹"，所以这里改为整框修复。
        atten = _overlay_attenuation(maps["tstd"], (x, y, w, h))
        if atten < atten_ratio:
            glyph = np.full(crop.shape, 255, np.uint8)
            dilate_extra = max(dilate_extra, 2)
            attenuated = True
        else:
            attenuated = False

        glyph = cv2.morphologyEx(glyph, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        glyph = _fill_holes(glyph)
        if dilate_extra > 0:
            glyph = cv2.dilate(
                glyph, np.ones((2 * dilate_extra + 1, 2 * dilate_extra + 1), np.uint8)
            )
        glyph_full = cv2.resize(glyph, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        region = binary[y0:y1, x0:x1]
        binary[y0:y1, x0:x1] = cv2.bitwise_or(region, glyph_full)

        per_box.append(
            {
                "id": cand.get("id"),
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "ink_ratio": round(float((glyph_full > 0).mean()), 4),
                "pad": [x0 - fx, y0 - fy],
                "ink_threshold": ink_thr,
                "dilate_extra": dilate_extra,
                "attenuation": round(float(atten), 3),
                "attenuated": attenuated,
            }
        )

    feather = cv2.GaussianBlur(binary.astype(np.float32) / 255.0, (0, 0), sigma)
    feather = np.clip(feather * 1.35, 0.0, 1.0)  # 羽化后略微加厚，防止边缘缝
    return {
        "binary": binary,
        "feather": feather.astype(np.float32),
        "regions": per_box,
        "coverage": round(float((binary > 0).mean()), 5),
        "sigma": sigma,
        "dilate_extra": dilate_extra,
        "ink_threshold": ink_thr,
    }


def mask_preview(binary: np.ndarray, feather: np.ndarray) -> np.ndarray:
    """把 mask 画成可读的预览图（白=mask，灰=羽化过渡）。"""
    heat = (feather * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    edge = cv2.dilate(binary, np.ones((3, 3), np.uint8))
    color[edge > 0] = (255, 255, 255)
    return color


# --------------------------------------------------------------------------- #
# 视频修复（逐帧单次解码 + ffmpeg 管道编码）
# --------------------------------------------------------------------------- #
def _ffmpeg_writer(
    dst: Path,
    size: tuple[int, int],
    fps: float,
    src: Path | None,
    has_audio: bool,
    crf: int = 18,
    preset: str = "veryfast",
) -> tuple[subprocess.Popen, Callable[[np.ndarray], None]]:
    """起一个 ffmpeg 进程，返回 (proc, write_frame)。没有 ffmpeg 时抛 VideoError。"""
    if not config.FFMPEG:
        raise VideoError("找不到 ffmpeg（runtime/ffmpeg/ffmpeg.exe 或 PATH）")
    width, height = size
    cmd = [
        config.FFMPEG,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{max(fps, 1.0):.6f}",
        "-i",
        "-",
    ]
    if src is not None:
        cmd += ["-i", str(src)]
        cmd += ["-map", "0:v:0"]
        if has_audio:
            cmd += ["-map", "1:a?", "-c:a", "copy"]
        else:
            cmd += ["-an"]
    cmd += [
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if src is not None and has_audio:
        cmd += ["-shortest"]
    cmd.append(str(dst))
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        creationflags=creationflags,
    )

    def write(frame: np.ndarray) -> None:
        if proc.stdin is None:
            raise VideoError("ffmpeg 管道已关闭")
        proc.stdin.write(frame.astype(np.uint8).tobytes())

    return proc, write


def _region_ring_masks(
    shape: tuple[int, int], regions: list[dict[str, Any]], ring_ratio: float = 0.35
) -> list[tuple[np.ndarray, np.ndarray]]:
    """为每个区域构造内部/邻域环带掩码（全分辨率）。"""
    height, width = shape
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for region in regions:
        x, y, w, h = [int(v) for v in region["bbox"]]
        inner = np.zeros((height, width), bool)
        inner[max(0, y) : min(height, y + h), max(0, x) : min(width, x + w)] = True
        pad_x = max(2, int(round(w * ring_ratio)))
        pad_y = max(2, int(round(h * ring_ratio)))
        outer = np.zeros((height, width), bool)
        outer[max(0, y - pad_y) : min(height, y + h + pad_y),
              max(0, x - pad_x) : min(width, x + w + pad_x)] = True
        ring = outer & ~inner
        if not ring.any():
            ring = ~inner
        out.append((inner, ring))
    return out


def measure_frame(
    gray: np.ndarray,
    ring_masks: list[tuple[np.ndarray, np.ndarray]],
    regions: list[dict[str, Any]],
    tex_sigma: float = 8.0,
) -> list[dict[str, float]]:
    """单帧的质量度量：纹理方差比、边缘密度比、亮度差。

    ``var_ratio`` 直接比整块方差，会被背景的大尺度渐变主导（同一块区域在
    缓变背景上方差可以有几倍差异，跟"有没有修糊"无关）。所以判定用的
    ``tex_ratio`` 比的是**高频纹理方差**：``hp = gray - blur(gray, σ)`` 后再比较，
    这才是 inpaint 会抹掉、而纹理回填要补回来的那部分能量。
    """
    gray_f = gray.astype(np.float32)
    hp = gray_f - cv2.GaussianBlur(gray_f, (0, 0), max(1.0, tex_sigma))
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150) > 0
    metrics: list[dict[str, float]] = []
    for (inner, ring), region in zip(ring_masks, regions):
        try:
            vals_in = gray[inner].astype(np.float32)
            vals_ring = gray[ring].astype(np.float32)
            if vals_in.size < 4 or vals_ring.size < 4:
                metrics.append({})
                continue
            var_in = float(vals_in.var())
            var_ring = float(vals_ring.var())
            var_ratio = abs(var_in - var_ring) / max(var_ring, 12.0)

            tex_in = float(hp[inner].var())
            tex_ring = float(hp[ring].var())
            tex_ratio = abs(tex_in - tex_ring) / max(tex_ring, 4.0)

            edge_in = float(edges[inner].mean())
            edge_ring = float(edges[ring].mean())
            edge_ratio = edge_in / max(edge_ring, 1e-6)

            mean_in = float(vals_in.mean())
            mean_ring = float(vals_ring.mean())
            std_ring = max(float(vals_ring.std()), 1.0)
            lum_diff = abs(mean_in - mean_ring) / std_ring

            metrics.append(
                {
                    "var_in": round(var_in, 2),
                    "var_ring": round(var_ring, 2),
                    "var_ratio": round(var_ratio, 4),
                    "tex_in": round(tex_in, 2),
                    "tex_ring": round(tex_ring, 2),
                    "tex_ratio": round(tex_ratio, 4),
                    "edge_in": round(edge_in, 5),
                    "edge_ring": round(edge_ring, 5),
                    "edge_ratio": round(edge_ratio, 4),
                    "lum_diff": round(lum_diff, 4),
                }
            )
        except Exception:  # pragma: no cover - 极端几何
            metrics.append({})
    return metrics


def _patch_fill(
    frame: np.ndarray,
    binary: np.ndarray,
    feather: np.ndarray,
    regions: list[dict[str, Any]],
    ring_ratio: float = 0.35,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """纹理移植：用画面里**真实存在**的邻近干净块填掉遮罩区（本项目自己的增量）。

    ``cv2.inpaint`` 只能做扩散，被修复区会变成一块"磨皮"色块：结构接上了，
    但纹理能量和运动都被抹平。这里换成 patch 思路：

    1. 对每个区域枚举几个候选源块（上下左右平移一个框高/框宽的距离）；
    2. 源块必须整体落在画面内、且不与遮罩区重叠（否则会把水印搬过来）；
    3. 用「源块外圈统计量 vs 目标区外圈统计量」的差异选最优（均值差 + 标准差差），
       再做一次均值平移，把亮度/色调对齐；
    4. 用羽化 alpha 软合成，接缝处不硬切。

    这样填进去的是同一段视频里真实的内容（有纹理、有运动、有噪声），
    质检的方差/纹理能量指标天然就能对上，不会像纯扩散那样一路偏平。
    """
    out = frame.astype(np.float32).copy()
    alpha = np.clip(feather.astype(np.float32), 0.0, 1.0)
    height, width = binary.shape[:2]
    info: list[dict[str, Any]] = []

    for region in regions:
        x, y, w, h = [int(v) for v in region["bbox"]]
        bx0, by0 = max(0, x), max(0, y)
        bx1, by1 = min(width, x + w), min(height, y + h)
        bw, bh = bx1 - bx0, by1 - by0
        if bw < 8 or bh < 8:
            info.append({"id": region.get("id"), "applied": False, "reason": "区域太小"})
            continue

        pad_x = max(4, int(round(bw * ring_ratio)))
        pad_y = max(4, int(round(bh * ring_ratio)))
        target_ring = _band_stats(frame, bx0, by0, bx1, by1, pad_x, pad_y)
        if target_ring is None:
            info.append({"id": region.get("id"), "applied": False, "reason": "环带不足"})
            continue

        gap = max(10, int(0.5 * min(bw, bh)))
        offsets = [
            (0, -(bh + gap)),
            (0, bh + gap),
            (-(bw + gap), 0),
            (bw + gap, 0),
            (-(bw + gap), -(bh + gap)),
            (bw + gap, bh + gap),
            (-(bw + gap), bh + gap),
            (bw + gap, -(bh + gap)),
        ]
        best: tuple[float, int, int] | None = None
        for dx, dy in offsets:
            sx0, sy0 = bx0 + dx, by0 + dy
            if sx0 < 0 or sy0 < 0 or sx0 + bw > width or sy0 + bh > height:
                continue
            if binary[sy0 : sy0 + bh, sx0 : sx0 + bw].any():
                continue
            source_ring = _band_stats(frame, sx0, sy0, sx0 + bw, sy0 + bh, pad_x, pad_y)
            if source_ring is None:
                continue
            score = _band_distance(target_ring, source_ring)
            if best is None or score < best[0]:
                best = (score, sx0, sy0)
        if best is None:
            info.append({"id": region.get("id"), "applied": False, "reason": "没有可用的干净源块"})
            continue

        _score, sx0, sy0 = best
        src_patch = frame[sy0 : sy0 + bh, sx0 : sx0 + bw].astype(np.float32)
        # 均值对齐：只搬纹理/结构，不改目标区的整体明暗
        shift = target_ring["mean"] - _band_stats(frame, sx0, sy0, sx0 + bw, sy0 + bh, pad_x, pad_y)["mean"]
        src_patch = src_patch + shift.reshape(1, 1, 3)
        a = alpha[by0:by1, bx0:bx1]
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, axis=2)
        region_view = out[by0:by1, bx0:bx1]
        out[by0:by1, bx0:bx1] = region_view * (1.0 - a) + src_patch * a
        info.append(
            {
                "id": region.get("id"),
                "applied": True,
                "offset": [sx0 - bx0, sy0 - by0],
                "score": round(float(_score), 3),
                "shift": [round(float(v), 3) for v in shift],
            }
        )
    return out, info


def _band_stats(
    frame: np.ndarray, x0: int, y0: int, x1: int, y1: int, pad_x: int, pad_y: int
) -> dict[str, np.ndarray] | None:
    """取矩形外侧一圈（pad 宽）的通道均值/标准差，用来比较两块区域的"环境"。"""
    height, width = frame.shape[:2]
    rx0, ry0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
    rx1, ry1 = min(width, x1 + pad_x), min(height, y1 + pad_y)
    patch = frame[ry0:ry1, rx0:rx1].astype(np.float32)
    if patch.size == 0:
        return None
    mask = np.ones(patch.shape[:2], bool)
    mask[max(0, y0 - ry0) : max(0, y1 - ry0), max(0, x0 - rx0) : max(0, x1 - rx0)] = False
    if mask.sum() < 32:
        return None
    values = patch[mask]
    return {"mean": values.mean(axis=0), "std": values.std(axis=0)}


def _band_distance(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    """两块区域的"环境相似度"：均值差按亮度尺度归一，再叠标准差差。"""
    mean_diff = np.abs(a["mean"] - b["mean"]).mean() / 12.0
    scale = np.maximum(a["std"].mean(), 4.0)
    std_diff = np.abs(a["std"] - b["std"]).mean() / scale
    return float(mean_diff + 0.5 * std_diff)


def _texture_restore(
    frame: np.ndarray,
    base: np.ndarray,
    binary: np.ndarray,
    feather: np.ndarray,
    regions: list[dict[str, Any]],
    sigma: float = 8.0,
    max_gain: float = 2.5,
    seed: int = 20241111,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """把被修复区域的"纹理能量"补回邻域水平（本项目自己的增量）。

    ``cv2.inpaint`` 只会做各向同性扩散，结果是一片"磨皮"的平滑色块：
    结构（渐变）勉强接上了，但纹理方差远低于邻域，肉眼就是一块糊斑。
    这里用「高频移植 + 方差配平」做二次补足：

    1. 取当前帧（水印还在，但我们只从**环带**采样）的高频分量 ``hp = frame - blur(frame, σ)``；
    2. 在被修复像素上按环带像素随机采样 ``hp``，得到一块统计特性与周围同源的纹理；
    3. 解一个一维方程 ``var_base + s²·var_hp = var_ring`` 求增益 ``s``，
       让补完后的区域方差对齐邻域（s 有上限，避免把噪点放大成雪花）；
    4. 依旧用羽化 alpha 软合成，所以边缘不会有硬接缝。

    返回 (restored_bgr_float32, 每个区域的诊断信息)。纹理增益会在 run 报告里留痕。
    """
    frame_f = frame.astype(np.float32)
    restored = base.astype(np.float32).copy()
    hp = frame_f - cv2.GaussianBlur(frame_f, (0, 0), max(1.0, sigma))
    base_hp = restored - cv2.GaussianBlur(restored, (0, 0), max(1.0, sigma))
    alpha = np.clip(feather.astype(np.float32), 0.0, 1.0)
    height, width = binary.shape[:2]
    rng = np.random.default_rng(seed)
    info: list[dict[str, Any]] = []

    for region in regions:
        x, y, rw, rh = [int(v) for v in region["bbox"]]
        # 环带取法必须与 measure_frame 一致（外扩 0.35 倍），否则补出来的能量对不上判定口径
        px, py = max(8, int(rw * 0.35)), max(8, int(rh * 0.35))
        x0, y0 = max(0, x - px), max(0, y - py)
        x1, y1 = min(width, x + rw + px), min(height, y + rh + py)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        win_binary = binary[y0:y1, x0:x1]
        win_alpha = alpha[y0:y1, x0:x1]
        ring = (win_binary == 0) & (win_alpha < 0.02)
        target = win_binary > 0
        if int(ring.sum()) < 64 or int(target.sum()) < 16:
            info.append({"id": region.get("id"), "applied": False, "reason": "环带/目标像素太少"})
            continue

        # bbox 可能被画面边界裁掉，先夹紧
        bx0, by0 = max(0, x), max(0, y)
        bx1, by1 = min(width, x + rw), min(height, y + rh)
        bw, bh = bx1 - bx0, by1 - by0
        if bw < 4 or bh < 4:
            continue
        target_bbox = binary[by0:by1, bx0:bx1] > 0
        if not target_bbox.any():
            continue

        hp_win = hp[y0:y1, x0:x1]
        base_hp_win = base_hp[y0:y1, x0:x1]
        # 判定口径同样是「高频纹理方差」，不是整块方差
        tex_ring = float(hp_win[ring].astype(np.float32).var())
        tex_base = float(base_hp_win[target].astype(np.float32).var())

        ys, xs = np.nonzero(ring)
        # 先在整块 bbox 上"铺满"环带高频采样，再极轻地模糊，
        # 让纹理成块而不是椒盐点（模糊太重会把高频能量削掉，只能靠增益硬拉）
        picks = rng.integers(0, len(ys), size=bh * bw)
        field = hp[y0:y1, x0:x1][ys[picks], xs[picks]].reshape(bh, bw, 3)
        field = cv2.GaussianBlur(field, (0, 0), max(1.2, min(bw, bh) / 24.0))
        field_vals = field[target_bbox].astype(np.float32)
        var_hp = float(field_vals.var())
        need = tex_ring - tex_base
        if var_hp <= 1e-3 or need <= 0:
            gain = 0.0
        else:
            gain = float(min(max_gain, math.sqrt(need / var_hp)))
        if gain > 0:
            restored[by0:by1, bx0:bx1][target_bbox] += field[target_bbox] * gain
        info.append(
            {
                "id": region.get("id"),
                "applied": gain > 0,
                "tex_ring": round(tex_ring, 2),
                "tex_base": round(tex_base, 2),
                "gain": round(gain, 3),
                "sigma": sigma,
            }
        )
    return restored, info


def inpaint_video(
    src: str | Path,
    dst: str | Path,
    masks: dict[str, Any],
    params: dict[str, Any],
    progress: Callable[[int, int, str], None] | None = None,
    metrics_every: int = 6,
) -> dict[str, Any]:
    """逐帧修复并编码。

    ``params``: ``method``（telea/ns/blur）、``radius``、``blur_mix``、``crf``。
    返回写出的帧数、耗时与**逐帧度量的聚合**（var_ratio/edge_ratio 的平均与最差），
    供 Refiner 判断"哪里还需要重修"。
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    meta = probe_video(src)
    width, height = meta["width"], meta["height"]
    fps = meta["fps"] or 25.0
    total = meta["frame_count"] or 0

    binary = masks["binary"]
    feather = masks["feather"]
    regions = masks.get("regions", [])
    if binary.shape[:2] != (height, width):
        binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
        feather = cv2.resize(feather, (width, height), interpolation=cv2.INTER_LINEAR)

    method = str(params.get("method", "telea")).lower()
    radius = int(params.get("radius", config.INPAINT_RADIUS))
    blur_mix = float(params.get("blur_mix", 0.0))
    crf = int(params.get("crf", config.VIDEO_CRF))
    # 纹理回填（本项目增量）：先把邻近真实块搬过来，再把残余的纹理能量补回邻域水平
    patch_fill = bool(params.get("patch_fill", True))
    texture = bool(params.get("texture", True))
    texture_sigma = float(params.get("texture_sigma", 8.0))
    texture_gain = float(params.get("texture_gain", 3.0))
    texture_stats: list[list[dict[str, Any]]] = []
    patch_stats: list[list[dict[str, Any]]] = []

    ring_masks = _region_ring_masks((height, width), regions) if regions else []
    alpha = feather[..., None] if feather.ndim == 2 else feather
    alpha_u8 = (np.clip(feather, 0.0, 1.0) * 255).astype(np.uint8)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise VideoError(f"无法打开视频：{src}")
    proc = None
    writer = cv2.VideoWriter()
    write_frame: Callable[[np.ndarray], None] | None = None
    use_ffmpeg = bool(config.FFMPEG)
    if use_ffmpeg:
        proc, write_frame = _ffmpeg_writer(
            dst, (width, height), fps, src, bool(meta.get("has_audio")), crf=crf
        )
    else:  # pragma: no cover - 无 ffmpeg 兜底
        writer.open(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise VideoError("既没有 ffmpeg，也无法用 OpenCV 写视频")

    stats: list[list[dict[str, float]]] = [[] for _ in regions]
    frames_written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out = frame
            if binary.any():
                if method == "blur":
                    patched = cv2.GaussianBlur(frame, (0, 0), max(3.0, radius * 1.6))
                else:
                    flags = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
                    patched = cv2.inpaint(frame, binary, radius, flags)
                if blur_mix > 0:
                    blurred = cv2.GaussianBlur(frame, (0, 0), max(3.0, radius * 1.8))
                    patched = cv2.addWeighted(patched, 1.0 - blur_mix, blurred, blur_mix, 0)
                restored = patched
                if patch_fill and regions:
                    restored, p_info = _patch_fill(frame, binary, feather, regions)
                    patch_stats.append(p_info)
                if texture and regions:
                    restored, tex_info = _texture_restore(
                        frame, restored, binary, feather, regions,
                        sigma=texture_sigma, max_gain=texture_gain,
                    )
                    texture_stats.append(tex_info)
                if alpha.ndim == 3:
                    a = alpha if alpha.shape[2] == 3 else np.repeat(alpha, 3, axis=2)
                    out = (
                        frame.astype(np.float32) * (1.0 - a)
                        + np.asarray(restored, np.float32) * a
                    )
                    out = np.clip(out, 0, 255).astype(np.uint8)
                else:
                    out = np.asarray(restored, np.uint8)

            if write_frame is not None:
                write_frame(out)
            else:  # pragma: no cover
                writer.write(out)
            frames_written += 1

            if regions and ring_masks and frames_written % max(1, metrics_every) == 0:
                gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                for i, item in enumerate(measure_frame(gray, ring_masks, regions)):
                    if item:
                        stats[i].append(item)

            if progress and (frames_written % 12 == 0 or frames_written == total):
                progress(frames_written, total or frames_written, f"已修复 {frames_written} 帧")
    finally:
        cap.release()
        if proc is not None:
            try:
                if proc.stdin:
                    proc.stdin.close()
                proc.wait(timeout=180)
            except subprocess.TimeoutExpired:  # pragma: no cover
                proc.kill()
            stderr = b""
            if proc.stderr:
                stderr = proc.stderr.read() or b""
                proc.stderr.close()
            if proc.returncode not in (0, None):
                raise VideoError(
                    "ffmpeg 编码失败：" + stderr.decode("utf-8", "replace")[-800:]
                )
        else:  # pragma: no cover
            writer.release()

    aggregated = []
    for i, region in enumerate(regions):
        rows = stats[i]
        if not rows:
            aggregated.append({"id": region.get("id"), "samples": 0})
            continue
        aggregated.append(
            {
                "id": region.get("id"),
                "samples": len(rows),
                "var_ratio_avg": round(float(np.mean([r["var_ratio"] for r in rows])), 4),
                "var_ratio_max": round(float(np.max([r["var_ratio"] for r in rows])), 4),
                "tex_ratio_avg": round(float(np.mean([r["tex_ratio"] for r in rows])), 4),
                "tex_ratio_max": round(float(np.max([r["tex_ratio"] for r in rows])), 4),
                "edge_ratio_avg": round(float(np.mean([r["edge_ratio"] for r in rows])), 4),
                "lum_diff_avg": round(float(np.mean([r["lum_diff"] for r in rows])), 4),
                "edge_ratio_max": round(float(np.max([r["edge_ratio"] for r in rows])), 4),
            }
        )

    patch_summary: list[dict[str, Any]] = []
    for i, region in enumerate(regions):
        rows = [row for frame_rows in patch_stats for row in (frame_rows or []) if row.get("id") == region.get("id")]
        offsets = [tuple(r["offset"]) for r in rows if r.get("offset")]
        best_offset = None
        if offsets:
            best_offset = max(set(offsets), key=offsets.count)
        patch_summary.append(
            {
                "id": region.get("id"),
                "samples": len(rows),
                "applied_frames": sum(1 for r in rows if r.get("applied")),
                "offset": list(best_offset) if best_offset else None,
            }
        )

    texture_summary: list[dict[str, Any]] = []
    for i, region in enumerate(regions):
        rows = [row for frame_rows in texture_stats for row in (frame_rows or []) if row.get("id") == region.get("id")]
        gains = [float(r["gain"]) for r in rows if "gain" in r]
        texture_summary.append(
            {
                "id": region.get("id"),
                "samples": len(rows),
                "gain_avg": round(float(np.mean(gains)), 3) if gains else 0.0,
                "applied_frames": sum(1 for r in rows if r.get("applied")),
            }
        )

    return {
        "frames_written": frames_written,
        "fps": fps,
        "size": [width, height],
        "duration": round(frames_written / max(fps, 1e-6), 2),
        "method": method,
        "radius": radius,
        "blur_mix": blur_mix,
        "patch": {
            "enabled": patch_fill,
            "regions": patch_summary,
        },
        "texture": {
            "enabled": texture,
            "sigma": texture_sigma,
            "gain_cap": texture_gain,
            "regions": texture_summary,
        },
        "regions": aggregated,
        "output": str(dst),
    }


# --------------------------------------------------------------------------- #
# 抽帧 / 对比图 / 校验
# --------------------------------------------------------------------------- #
def read_frames_at(path: str | Path, indices: Iterable[int]) -> list[np.ndarray]:
    """按帧号读原图（质检抽帧）。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(f"无法打开视频：{path}")
    out: list[np.ndarray] = []
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ok, frame = cap.read()
            if ok:
                out.append(frame)
    finally:
        cap.release()
    return out


def evenly_spaced_indices(total: int, count: int) -> list[int]:
    """均匀抽帧（避开首尾各 5%，减少黑场/片尾）。"""
    if total <= 0:
        return [0] * count
    lo = int(total * 0.05)
    hi = max(lo + 1, int(total * 0.95))
    count = max(1, count)
    return sorted({int(round(v)) for v in np.linspace(lo, hi - 1, count)})


def crop_with_context(frame: np.ndarray, bbox: Sequence[int], context: float = 1.6) -> np.ndarray:
    """按 bbox 裁一块带上下文的正方形区域（质检看图用）。"""
    height, width = frame.shape[:2]
    x, y, w, h = [int(v) for v in bbox]
    side = int(max(w, h) * context)
    side = max(side, 32)
    cx, cy = x + w // 2, y + h // 2
    x0 = max(0, cx - side // 2)
    y0 = max(0, cy - side // 2)
    x1 = min(width, x0 + side)
    y1 = min(height, y0 + side)
    x0 = max(0, x1 - side)
    y0 = max(0, y1 - side)
    return frame[y0:y1, x0:x1].copy()


def side_by_side(panels: list[np.ndarray], labels: list[str], tile_height: int = 220) -> np.ndarray:
    """多块面板横排 + 顶部标签条（BGR）。"""
    prepared: list[np.ndarray] = []
    for panel in panels:
        if panel is None or panel.size == 0:
            panel = np.zeros((tile_height, tile_height, 3), np.uint8)
        scale = tile_height / float(panel.shape[0])
        resized = cv2.resize(panel, (max(1, int(panel.shape[1] * scale)), tile_height))
        prepared.append(resized)
    width = sum(p.shape[1] for p in prepared)
    band = 26
    canvas = np.full((tile_height + band, width, 3), 24, np.uint8)
    cursor = 0
    for panel, label in zip(prepared, labels):
        canvas[band : band + tile_height, cursor : cursor + panel.shape[1]] = panel
        cv2.putText(
            canvas,
            label,
            (cursor + 6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        cv2.line(
            canvas,
            (cursor, band),
            (cursor + panel.shape[1], band),
            (70, 70, 70),
            1,
        )
        cursor += panel.shape[1]
    return canvas


def annotate_boxes(frame: np.ndarray, candidates: list[dict[str, Any]], scale: float) -> np.ndarray:
    """在帧上画出检测框+置信度（报告配图）。"""
    out = frame.copy()
    for cand in candidates:
        x, y, w, h = [int(round(v / max(scale, 1e-6))) for v in cand["bbox"]]
        color = (60, 220, 120) if cand["confidence"] >= 0.6 else (60, 180, 250)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        text = f"{cand.get('id', '')} {cand['confidence']:.2f}"
        cv2.putText(
            out, text, (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )
    return out


def write_image(path: str | Path, image: np.ndarray) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise VideoError(f"图片写入失败：{path}")
    path.write_bytes(buf.tobytes())
    return str(path)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))