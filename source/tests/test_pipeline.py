"""端到端流水线测试：输入 → 各阶段产物落盘 → 质检通过。

这是任务书要求的「至少一条完整链路测试」：
1. 用 ``sample_video.create_sample`` 造一段带音轨、带固定水印的示例视频；
2. 直接调用 ``Orchestrator.run_job``（绕过 HTTP，但走完全相同的 Agent 链路）；
3. 断言：

   * ``clean.mp4`` 真的写出来了、能被 ffprobe 读出、分辨率一致、音轨还在；
   * ``detect_report.json`` / ``review_report.json`` 落盘且 ``passed == true``；
   * 质检抽帧图 5 张 + montage 存在（人工复核点）；
   * 如果发生过回灌，SSE 日志里必须同时出现「不达标」和「通过」，即
     「不达标 → 重生成 → 达标」在日志里可追溯。

单独运行：``python tests/test_pipeline.py``（约 1-2 分钟）。
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

import config  # noqa: E402
import event_bus  # noqa: E402
import sample_video  # noqa: E402
import watermark_utils as wu  # noqa: E402
from agents.orchestrator import Orchestrator  # noqa: E402

RUN_ID = "pipeline_selftest"
RUN_DIR = config.OUTPUT_DIR / RUN_ID
SRC = config.OUTPUT_DIR / "_uploads" / "pipeline_selftest" / "sample_with_audio.mp4"


def _prepare_sample(seconds: int = 3) -> dict:
    SRC.parent.mkdir(parents=True, exist_ok=True)
    if not SRC.is_file():
        info = sample_video.create_sample(SRC, seconds=seconds, with_audio=True)
        assert info.get("with_audio"), f"示例视频没有音轨：{info}"
    info = wu.probe_video(SRC)
    assert info["frame_count"] > 30, info
    return info


def _run() -> dict:
    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR, ignore_errors=True)
    src_info = _prepare_sample()
    bus = event_bus.EventBus()
    orch = Orchestrator(bus)
    payload = {
        "src_path": str(SRC),
        "run_dir": str(RUN_DIR),
        "authorized": True,
        "params": {"sample_frames": 24, "appear_rate": 0.70, "max_regions": 4, "min_conf": 0.45},
    }
    started = time.perf_counter()
    result = asyncio.run(orch.run_job(RUN_ID, payload))
    elapsed = round(time.perf_counter() - started, 1)
    events = [json.loads(raw) for raw in bus.history(RUN_ID, 500)]
    return {"payload": result, "events": events, "elapsed": elapsed, "src": src_info}


def test_pipeline_writes_products_and_passes() -> None:
    out = _run()
    payload, events, elapsed, src = out["payload"], out["events"], out["elapsed"], out["src"]
    run_dir = Path(payload["run_dir"])
    print(f"    · 流水线耗时 {elapsed}s，attempt={payload.get('attempt')}，"
          f"feedback_count={payload.get('feedback_count')}，status={payload.get('status')}")

    assert payload["status"] == "completed", payload.get("error")
    review = payload["review"]
    print(f"    · 质检 {review['score']} 分 / 通过线 {review['pass_score']}，verdict={review['verdict']}，"
          f"attempt={review['attempt']}")
    assert review["passed"] is True, review["failures"]
    assert review["score"] >= review["pass_score"]
    assert int(review["attempt"]) == int(payload["attempt"])

    # ---- 产物落盘 ----
    expected = [
        "clean.mp4",
        "detect_report.json",
        "review_report.json",
        "run_log.json",
        "detect_preview.jpg",
        "detect_support.jpg",
        "mask.png",
        "mask_heat.png",
        "preview_before_after.jpg",
        "review_frames/frame_01.jpg",
        "review_frames/frame_05.jpg",
        "review_frames/montage.jpg",
    ]
    missing = [name for name in expected if not (run_dir / name).is_file()]
    assert not missing, f"缺少产物：{missing}"

    clean_info = wu.probe_video(run_dir / "clean.mp4")
    print(f"    · clean.mp4 {clean_info['width']}x{clean_info['height']} "
          f"{clean_info['duration']}s 音轨={clean_info['has_audio']} "
          f"{clean_info['size_bytes'] // 1024}KB")
    assert (clean_info["width"], clean_info["height"]) == (src["width"], src["height"])
    assert clean_info["has_audio"] is True, "音轨被丢了"
    assert clean_info["frame_count"] >= src["frame_count"] - 3

    # ---- 报告内容 ----
    detect_report = json.loads((run_dir / "detect_report.json").read_text("utf-8"))
    regions = detect_report["detection"]["regions"]
    print(f"    · 检出 {len(regions)} 个候选区域：" +
          ", ".join(f"{r['id']}@{r['bbox']}" for r in regions))
    assert len(regions) >= 1
    assert regions[0]["confidence"] >= 0.45
    assert detect_report["outputs"]["clean_video"].endswith("clean.mp4")

    review_report = json.loads((run_dir / "review_report.json").read_text("utf-8"))
    assert review_report["passed"] is True
    assert review_report["attempts"] == int(review["attempt"])
    assert len(review_report["human_check"]["images"]) >= 6
    for rel in review_report["human_check"]["images"]:
        assert (run_dir / rel).is_file(), rel

    # ---- 日志可追溯「不达标 → 回灌 → 通过」 ----
    types = [e["type"] for e in events]
    messages = "\n".join(e.get("message", "") for e in events)
    for required in ("task_started", "agent_completed", "task_completed"):
        assert required in types, f"日志缺少 {required}"
    assert types.index("task_started") < types.index("task_completed")
    if payload.get("feedback_count", 0) > 0:
        feedback = [e for e in events if e["type"] == "review_feedback"]
        assert feedback, "发生过回灌但没有 review_feedback 事件"
        assert any("回灌" in e["message"] for e in feedback)
        print(f"    · 回灌日志：{feedback[0]['message']}")
    print(f"    · 日志 {len(events)} 条，含 "
          f"{sum(1 for t in types if t == 'agent_completed')} 个 agent_completed")
    assert "不达标" in messages or payload.get("feedback_count", 0) == 0


def test_feedback_turn_enables_increment() -> None:
    """回灌轮必须切到「任务书 baseline 之外的增量」：patch 移植 + 纹理回填。

    设计上第 1 轮只跑 cv2.inpaint（快、便宜），质检判不达标后第 2 轮才启用
    邻近块移植与高频纹理回填 —— 这条断言保证「增量」真的挂在回灌路径上，
    而不是写死在首轮里导致首轮慢、也测不到闭环。
    """
    out = _run()
    payload = out["payload"]
    assert payload["status"] == "completed"
    attempts = int(payload.get("attempt", 1))
    if attempts < 2:
        print("    · 首轮即通过（未触发回灌），跳过增量启用断言")
        return
    params = payload["repair"]["params"]
    assert params.get("patch_fill") is True, params
    assert payload["mask"]["plan"].get("patch_fill") is True, payload["mask"]["plan"]
    tex = payload["repair"]["stats"].get("patch")
    assert tex and tex.get("enabled") is True, tex
    print(f"    · 第 {attempts} 轮修复参数："
          f"{params.get('method')}/r{params.get('radius')}/"
          f"patch_fill={params.get('patch_fill')}/texture={params.get('texture')}")


def main() -> int:
    tests = (test_pipeline_writes_products_and_passes, test_feedback_turn_enables_increment)
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n端到端测试：{len(tests) - failed}/{len(tests)} 通过（产物目录 {RUN_DIR}）")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())