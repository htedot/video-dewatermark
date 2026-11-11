"""契约测试：四件套（BaseAgent / EventBus / SSE / LLM）与硬性工程约束。

不依赖 pytest，直接 ``python tests/test_contracts.py`` 跑；每个 test_* 函数
要么返回 None（通过），要么抛 AssertionError（失败）。

这些测试对应的都是任务书里最容易踩的坑：
* ``bus.subscribe()`` 必须是同步方法（写成 async 会让 SSE 生成器拿不到队列）；
* ``bus.publish()`` 内部必须 json.dumps（上层只拿字符串，SSE 直接拼接）；
* LLM 返回的 JSON 必须走 ``_extract_json`` 平衡括号扫描，不能直接 ``json.loads``；
* Agent 之间禁止互相 import（只能由 orchestrator 串起来）；
* .bat 必须纯 ASCII + CRLF + 无 BOM（否则 cmd 解析报错）；
* 端口必须在 8200-8299，且 ffmpeg/ffprobe 真能跑。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

import config  # noqa: E402
import event_bus  # noqa: E402
import llm as llm_module  # noqa: E402
import watermark_utils as wu  # noqa: E402
from agents.base import BaseAgent  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. EventBus / SSE 契约
# --------------------------------------------------------------------------- #
def test_bus_subscribe_is_sync() -> None:
    """subscribe() 必须是同步方法：异步化会让 SSE 拿不到队列。"""
    assert not inspect.iscoroutinefunction(event_bus.EventBus.subscribe)
    bus = event_bus.EventBus()
    queue = bus.subscribe()
    assert isinstance(queue, asyncio.Queue), type(queue)
    bus.unsubscribe(queue)


def test_bus_publish_returns_json_string() -> None:
    """publish() 内部 json.dumps，返回可直接喂 SSE 的字符串。"""
    bus = event_bus.EventBus()
    raw = bus.publish({"type": "agent_progress", "run_id": "r1", "message": "中文消息"})
    assert isinstance(raw, str), type(raw)
    data = json.loads(raw)  # 能反解，说明是合法 JSON
    assert data["type"] == "agent_progress"
    assert data["run_id"] == "r1"
    assert data["message"] == "中文消息"
    assert "seq" in data and "ts" in data
    # 中文不应被转义成 \uXXXX（观感 + 体积）
    assert "中文消息" in raw


def test_sse_format() -> None:
    raw = json.dumps({"type": "x"}, ensure_ascii=False)
    assert event_bus.format_sse(raw) == f"data: {raw}\n\n"


async def _bus_roundtrip() -> None:
    bus = event_bus.EventBus()
    queue = bus.subscribe()
    bus.emit("agent_started", "runA", "A 开始", agent="mask")
    await asyncio.sleep(0.02)
    evt = json.loads(queue.get_nowait())
    assert evt["run_id"] == "runA"
    assert evt["data"]["agent"] == "mask"
    # 事件按 run_id 隔离：history 只回放该 run（返回的是 JSON 字符串）
    bus.emit("agent_started", "runB", "B 开始")
    await asyncio.sleep(0.02)
    assert [json.loads(raw)["run_id"] for raw in bus.history("runA")] == ["runA"]
    assert all(json.loads(raw)["run_id"] == "runB" for raw in bus.history("runB"))
    bus.unsubscribe(queue)


def test_bus_run_id_isolation() -> None:
    asyncio.run(_bus_roundtrip())


# --------------------------------------------------------------------------- #
# 2. LLM 契约：JSON 提取 / dict vs list / 超时降级
# --------------------------------------------------------------------------- #
def test_extract_json_variants() -> None:
    fenced = '好的，结果如下：\n```json\n{"selected": [{"id": "c1"}], "note": "含 } 与 { 的字符串"}\n```\n以上。'
    data = llm_module._extract_json(fenced)
    assert isinstance(data, dict), data
    assert data["selected"][0]["id"] == "c1"
    assert "含 } 与 { 的字符串" in data["note"]

    listed = 'prefix [{"a": 1}, {"b": [2, 3]}] suffix'
    data = llm_module._extract_json(listed)
    assert isinstance(data, list) and data[1]["b"] == [2, 3]

    assert llm_module._extract_json("没有 JSON 的一段话") is None
    assert llm_module._extract_json("") is None
    # 截断的 JSON 不能抛异常，只能返回 None
    assert llm_module._extract_json('{"a": [1, 2') is None


class _FakeLLM(llm_module.LLMClient):
    """把网络调用换掉，专测「返回 JSON 字符串 → 解析」这一段契约。"""

    def __init__(self, payload: str) -> None:
        super().__init__(api_key="test-key", base_url="http://127.0.0.1:1/v1", model="fake")
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

    async def _chat(self, system, user, max_tokens=None, temperature=None, json_mode=False):  # type: ignore[override]
        self.calls.append((system, user))
        return self.payload, 17


def test_llm_complete_json_dict_only() -> None:
    client = _FakeLLM('前言 {"ok": true} 后语')
    data, tokens = asyncio.run(client.complete_json("sys", "user"))
    assert isinstance(data, dict) and data["ok"] is True
    assert tokens == 17

    # 顶层是 list 时，complete_json 必须报错（契约：list 走 complete_json_list）
    client = _FakeLLM('[{"a": 1}]')
    try:
        asyncio.run(client.complete_json("sys", "user"))
    except llm_module.LLMError:
        pass
    else:  # pragma: no cover
        raise AssertionError("complete_json 收到 list 应当抛 LLMError")


def test_llm_complete_json_list() -> None:
    client = _FakeLLM('```json\n[{"id": "c1"}, {"id": "c2"}]\n```')
    data, _ = asyncio.run(client.complete_json_list("sys", "user"))
    assert isinstance(data, list) and [d["id"] for d in data] == ["c1", "c2"]


def test_llm_complete_text_returns_tuple() -> None:
    client = _FakeLLM("一句评语")
    text, tokens = asyncio.run(client.complete_text("sys", "user"))
    assert text == "一句评语" and isinstance(tokens, int)


class _DummyAgent(BaseAgent):
    name = "dummy"
    label = "Dummy"

    async def run(self, payload):  # pragma: no cover - 只为实例化
        return payload


def test_agent_call_llm_degrades_without_key() -> None:
    """没有 Key 时 call_llm 必须返回 (None, 0)，绝不抛异常（否则流水线断）。"""
    bus = event_bus.EventBus()
    client = llm_module.LLMClient(api_key="", base_url="", model="m")
    agent = _DummyAgent(bus=bus, llm=client)
    result, tokens = asyncio.run(
        agent.call_llm({"run_id": "r-degrade"}, "unit", "sys", "user")
    )
    assert result is None and tokens == 0


# --------------------------------------------------------------------------- #
# 3. Agent 解耦 / 编排层
# --------------------------------------------------------------------------- #
def test_agents_do_not_import_each_other() -> None:
    """业务 Agent 之间禁止互相 import，只能通过 payload 传递数据。"""
    agents_dir = BACKEND / "agents"
    offenders = []
    for path in agents_dir.glob("*_agent.py"):
        text = path.read_text("utf-8")
        for match in re.finditer(r"^\s*(?:from|import)\s+agents\.(\w+)", text, re.M):
            if match.group(1) not in {"base"}:
                offenders.append(f"{path.name} -> agents.{match.group(1)}")
    assert not offenders, offenders
    # orchestrator 是唯一允许 import 全部业务 Agent 的地方
    orch = (agents_dir / "orchestrator.py").read_text("utf-8")
    assert "from agents.detector_agent import" in orch


def test_orchestrator_layers() -> None:
    from agents.orchestrator import Orchestrator

    orch = Orchestrator(event_bus.EventBus())
    layers = [list(layer) for layer in orch.layers]
    assert layers[0] == ["detector"] and layers[1] == ["mask"]
    assert layers[2] == ["inpainter"] and layers[3] == ["refiner"] and layers[4] == ["reviewer"]
    assert set(layers[5]) == {"preview", "report"}, layers[5]
    # 质检回灌边必须指向 mask
    assert orch.feedback_target == "mask"


# --------------------------------------------------------------------------- #
# 4. 工程约束：端口 / 二进制 / .bat
# --------------------------------------------------------------------------- #
def test_port_in_range() -> None:
    assert 8200 <= config.PORT <= 8299, config.PORT


def test_binaries_available() -> None:
    assert config.FFMPEG, "找不到 ffmpeg"
    assert config.FFPROBE, "找不到 ffprobe"
    assert Path(config.FFMPEG).is_file()
    assert Path(config.FFPROBE).is_file()


def test_bat_files_ascii_crlf_no_bom() -> None:
    for name in ("start.bat", "stop.bat", "run_tests.bat"):
        path = config.PROJECT_DIR / name
        assert path.is_file(), f"{name} 缺失"
        raw = path.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), f"{name} 带 BOM"
        assert raw.isascii(), f"{name} 含非 ASCII 字符"
        assert b"\r\n" in raw, f"{name} 不是 CRLF 换行"
        assert b"\n" not in raw.replace(b"\r\n", b""), f"{name} 含裸 LF"


# --------------------------------------------------------------------------- #
# 5. 算法层：候选生成 / 遮罩羽化 / 质检指标
# --------------------------------------------------------------------------- #
def test_generate_candidates_finds_static_box() -> None:
    import cv2
    import numpy as np

    rng = np.random.default_rng(7)
    frames = []
    for _ in range(12):
        base = np.full((240, 320), 120, np.uint8)
        base = cv2.GaussianBlur(base + rng.normal(0, 8, base.shape), (0, 0), 2)
        # 每帧移动的干扰物（不应被当成静态水印）
        moving = base.copy()
        cv2.circle(moving, (int(40 + rng.integers(0, 240)), 120), 12, 250, -1)
        frames.append(moving.astype(np.uint8))
    for i, frame in enumerate(frames):
        stamp = frame.copy()
        cv2.putText(stamp, "DEMO", (200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 20, 2, cv2.LINE_AA)
        frames[i] = stamp
    grays = np.stack(frames)
    maps = wu.compute_static_maps(grays)
    presets = json.loads((config.DATA_DIR / "watermark_presets.json").read_text("utf-8"))
    cands = wu.generate_candidates(maps, presets, 0.7)
    assert cands, "没有检出静态水印"
    best = max(cands, key=lambda c: c["confidence"])
    x, y, w, h = best["bbox"]
    assert 180 <= x <= 320 and 10 <= y <= 60, best["bbox"]  # 命中左上那块文字
    for c in cands:
        assert 0.0 <= c["confidence"] <= 1.0
        assert c["appear_rate"] >= 0.7, c


def test_build_masks_feather_and_coverage() -> None:
    import cv2
    import numpy as np

    rng = np.random.default_rng(3)
    grays = []
    for _ in range(10):
        frame = np.full((180, 240), 110, np.uint8)
        frame = cv2.GaussianBlur(frame + rng.normal(0, 6, frame.shape), (0, 0), 2)
        cv2.rectangle(frame, (30, 30), (120, 60), 30, -1)
        grays.append(frame.astype(np.uint8))
    maps = wu.compute_static_maps(np.stack(grays))
    presets = json.loads((config.DATA_DIR / "watermark_presets.json").read_text("utf-8"))
    cands = wu.generate_candidates(maps, presets, 0.7)
    assert cands
    masks = wu.build_masks(maps, cands[:1], (240, 180), 1.0,
                           {"ink_threshold": 0.5, "dilate_extra": 2, "pad_ratio": 0.1,
                            "feather_sigma": 7.0, "attenuation_ratio": 0.85})
    binary, feather = masks["binary"], masks["feather"]
    assert binary.shape == (180, 240)
    assert binary.sum() > 0
    assert feather.dtype == np.float32 and feather.max() <= 1.0
    # 羽化边必须比二值 mask 更软：0 < feather < 1 的像素要实际存在
    soft = ((feather > 0.02) & (feather < 0.98)).sum()
    assert soft > 50, soft
    assert masks["coverage"] > 0


def test_measure_frame_identical_is_perfect() -> None:
    """同一张纹理图里，区域与环带的纹理/边缘/亮度三项都应判为「接续」。"""
    import numpy as np

    h, w = 160, 200
    yy, xx = np.mgrid[0:h, 0:w]
    gray = (
        128 + 40 * np.sin(2 * np.pi * xx / 6) + 40 * np.sin(2 * np.pi * yy / 6)
    ).clip(0, 255).astype(np.uint8)
    regions = [{"id": "c1", "bbox": [60, 40, 80, 60]}]
    ring = wu._region_ring_masks(gray.shape, regions, 0.35)
    metric = wu.measure_frame(gray, ring, regions)[0]
    # 同一套纹理 → 三项指标都该落在「接续」范围
    assert metric["tex_ratio"] < 0.15, metric
    assert 0.7 < metric["edge_ratio"] < 1.4, metric
    assert metric["lum_diff"] < 0.2, metric


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def main() -> int:
    tests = [
        obj
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
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
    print(f"\n契约测试：{len(tests) - failed}/{len(tests)} 通过")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())