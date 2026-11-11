"""生成一段「带静态水印的示例视频」，方便没有素材时直接试跑/写测试。

画面内容每帧都在动（渐变底 + 平滑噪声 + 漂移圆 + 滚动条码 + 游走的场景号），
只有两处固定叠加层每帧完全一致：

* 右上角文字 「DEMO © SAMPLE 2024」（半透明白 + 描边）
* 左下角圆角徽章 「PREVIEW」

所以多帧边缘投票会把这两块投成高出现率区域，正好用来验证检测/遮罩/修复。
注意：底图故意做成"每帧都不同"的噪声+运动，避免静态背景干扰投票。
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import config

_SAMPLE_W, _SAMPLE_H, _SAMPLE_FPS = 1280, 720, 30


def _background(t: float, w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    """随时间变化的底图：竖直渐变 + 平滑噪声 + 两个漂移圆 + 滚动条码。"""
    ys = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    xs = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    frame = np.zeros((h, w, 3), np.uint8)
    frame[:, :, 0] = (40 + 50 * ys + 18 * np.sin(xs * 4 + t)).astype(np.uint8)
    frame[:, :, 1] = (70 + 80 * (1 - ys) + 20 * xs).astype(np.uint8)
    frame[:, :, 2] = (150 - 55 * xs + 15 * np.cos(ys * 3 + t * 0.8)).astype(np.uint8)

    # 平滑噪声：每帧重画，制造"到处都是弱边缘但都不稳定"的真实感底噪
    noise = rng.normal(0, 26, (h // 8, w // 8)).astype(np.float32)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    frame = np.clip(frame.astype(np.float32) + noise[:, :, None] * np.array([1.1, 0.9, 0.8]), 0, 255).astype(np.uint8)

    # 漂移圆
    cx1 = int(w * (0.25 + 0.15 * math.sin(t * 1.3)))
    cy1 = int(h * (0.45 + 0.18 * math.cos(t * 1.1)))
    cv2.circle(frame, (cx1, cy1), 90, (230, 200, 60), -1, cv2.LINE_AA)
    cx2 = int(w * (0.72 + 0.12 * math.cos(t * 0.9)))
    cy2 = int(h * (0.30 + 0.22 * math.sin(t * 1.7)))
    cv2.circle(frame, (cx2, cy2), 60, (60, 220, 140), -1, cv2.LINE_AA)

    # 滚动条码
    bar_y = int((t * 180) % (h + 80)) - 40
    cv2.rectangle(frame, (0, bar_y), (w, bar_y + 24), (90, 90, 90), -1)
    for i in range(0, w, 48):
        if (i // 48 + int(t * 8)) % 2 == 0:
            cv2.rectangle(frame, (i, bar_y), (i + 22, bar_y + 24), (200, 200, 200), -1)

    # 场景号：位置也在动，避免变成第二块"静态文字"
    sx = int(w * 0.5 + 220 * math.sin(t * 0.9))
    sy = int(h * 0.62 + 60 * math.cos(t * 1.4))
    cv2.putText(
        frame, f"scenario {int(t * _SAMPLE_FPS):05d}", (sx - 140, sy),
        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (250, 250, 250), 2, cv2.LINE_AA,
    )
    return frame


def _draw_watermarks(frame: np.ndarray, t: float) -> np.ndarray:
    """固定位置、固定像素的叠加层（每帧完全一致 → 出现率 = 1.0）。

    只让一个极小的呼吸亮度变化（±2 灰阶）模拟"叠加层被视频编码轻微抖动"，
    不改变几何位置，从而仍然满足"静态水印"的定义。
    """
    h, w = frame.shape[:2]
    out = frame.copy()

    # ---- 右上角文字水印 ----
    text = "DEMO © SAMPLE 2024"
    font = cv2.FONT_HERSHEY_DUPLEX
    scale, thick = 0.95, 2
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    x, y = w - tw - 48, 56
    dim = 0.80 + 0.02 * math.sin(t * 4.0)  # 只是一点点亮度呼吸，位置不动
    roi = out[y - th - 10 : y + base + 8, x - 12 : x + tw + 12]
    overlay = roi.copy()
    cv2.putText(overlay, text, (12, th + 10), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(overlay, text, (12, th + 10), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    out[y - th - 10 : y + base + 8, x - 12 : x + tw + 12] = cv2.addWeighted(
        roi, 1 - dim, overlay, dim, 0
    )

    # ---- 左下角圆角徽章 ----
    bw, bh = 176, 52
    bx, by = 40, h - bh - 40
    badge = out.copy()
    cv2.rectangle(badge, (bx, by), (bx + bw, by + bh), (20, 20, 20), -1)
    cv2.putText(
        badge, "PREVIEW", (bx + 18, by + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
        (235, 235, 235), 2, cv2.LINE_AA,
    )
    out = cv2.addWeighted(out, 0.45, badge, 0.55, 0)
    cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (210, 210, 210), 2)
    return out


def _ffmpeg_writer(path: Path, with_audio: bool):
    """优先用 ffmpeg 写 H.264（真实编码链路）；没有就退化到 cv2.VideoWriter。"""
    cmd = [
        config.FFMPEG or "ffmpeg",
        "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{_SAMPLE_W}x{_SAMPLE_H}", "-r", str(_SAMPLE_FPS),
        "-i", "pipe:0",
    ]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
                "-shortest", "-c:a", "aac", "-b:a", "96k"]
    else:
        cmd += ["-an"]
    cmd += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    def write(frame: np.ndarray) -> None:
        assert proc.stdin is not None
        proc.stdin.write(frame.tobytes())

    return proc, write


def create_sample(dest: Path | str, seconds: int = 6, with_audio: bool = False) -> dict[str, Any]:
    """渲染并编码示例视频，返回说明信息。"""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = max(1, int(seconds) * _SAMPLE_FPS)
    used_ffmpeg = bool(config.FFMPEG) and Path(config.FFMPEG).is_file()
    rng = np.random.default_rng(20241111)

    if used_ffmpeg:
        proc, write = _ffmpeg_writer(dest, with_audio)
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(dest), fourcc, _SAMPLE_FPS, (_SAMPLE_W, _SAMPLE_H))
        if not vw.isOpened():
            raise RuntimeError("既没有 ffmpeg，也打不开 cv2.VideoWriter")
        write = vw.write  # type: ignore[assignment]

    try:
        for i in range(total):
            t = i / _SAMPLE_FPS
            frame = _background(t, _SAMPLE_W, _SAMPLE_H, rng)
            frame = _draw_watermarks(frame, t)
            write(frame)
    finally:
        if used_ffmpeg:
            assert proc.stdin is not None
            proc.stdin.close()
            err = proc.stderr.read().decode("utf-8", "ignore") if proc.stderr else ""
            code = proc.wait(timeout=120)
            if code != 0:
                raise RuntimeError(f"ffmpeg 编码示例视频失败：{err[-400:]}")
        else:
            vw.release()

    return {
        "path": str(dest),
        "width": _SAMPLE_W,
        "height": _SAMPLE_H,
        "fps": _SAMPLE_FPS,
        "frames": total,
        "seconds": seconds,
        "with_audio": with_audio and used_ffmpeg,
        "note": "示例视频：底图持续运动，右上角文字与左下角徽章为固定叠加层（出现率 100%）",
    }


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sample_watermark.mp4")
    print(create_sample(target, int(sys.argv[2]) if len(sys.argv) > 2 else 6))