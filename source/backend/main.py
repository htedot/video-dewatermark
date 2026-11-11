"""FastAPI 入口：挂前端、发 SSE、收上传、起流水线、取产物。

路由一览
--------
GET  /                          前端页面（index.html）
GET  /api/health                健康检查（含 ffmpeg / LLM 状态）
GET  /api/dag                   DAG 描述（节点、依赖、并行层、回灌边）
GET  /api/events?run_id=...     SSE 事件流（按 run_id 隔离）
POST /api/upload                上传视频，返回 input_path
POST /api/sample                生成一段带水印的示例视频（无素材时用）
POST /api/generate              启动一次去水印任务（必须 authorized=true）
GET  /api/runs                  历史任务列表
GET  /api/runs/{run_id}         单任务状态 + 两份报告 + 事件回放
GET  /api/runs/{run_id}/files/{path}  取产物文件
GET  /api/runs/{run_id}/video   取 clean.mp4（给 <video> 用）
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import config
import llm as llm_module
import sample_video
import watermark_utils as wu
from agents.orchestrator import Orchestrator
from event_bus import bus

app = FastAPI(title=config.APP_NAME, version=config.APP_VERSION)
orchestrator = Orchestrator(bus)

# run_id -> 内存态（进程重启后靠扫 outputs/ 目录兜底）
JOBS: dict[str, dict[str, Any]] = {}
TASKS: dict[str, asyncio.Task] = {}
_STARTED_AT = time.time()
_RUN_ID_RE = re.compile(config.RUN_ID_PATTERN)


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _now() -> float:
    return round(time.time(), 3)


def _ffmpeg_status() -> dict[str, Any]:
    path = config.FFMPEG
    ok = False
    version = ""
    if path:
        try:
            import subprocess

            proc = subprocess.run(
                [path, "-hide_banner", "-version"],
                capture_output=True,
                timeout=8,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            ok = proc.returncode == 0
            if ok:
                version = proc.stdout.decode("utf-8", "ignore").splitlines()[0][:120]
        except Exception as exc:  # noqa: BLE001
            version = f"不可用：{exc}"
    return {"path": path, "ok": ok, "version": version, "ffprobe": config.FFPROBE}


def _safe_run_file(run_id: str, rel: str) -> Path:
    if not _RUN_ID_RE.match(run_id):
        raise HTTPException(400, "run_id 不合法")
    run_dir = (config.OUTPUT_DIR / run_id).resolve()
    target = (run_dir / rel).resolve()
    if not str(target).startswith(str(run_dir)):
        raise HTTPException(403, "路径越界")
    if not target.is_file():
        raise HTTPException(404, f"文件不存在：{rel}")
    return target


def _scan_runs() -> list[str]:
    if not config.OUTPUT_DIR.is_dir():
        return []
    out = []
    for d in config.OUTPUT_DIR.iterdir():
        if d.is_dir() and _RUN_ID_RE.match(d.name) and (d / "detect_report.json").exists():
            out.append(d.name)
        elif d.is_dir() and _RUN_ID_RE.match(d.name):
            out.append(d.name)
    return sorted(set(out))


def _basename(path: str | None) -> str | None:
    """从绝对路径里取文件名；服务重启后历史任务靠它回填来源文件名。"""
    if not path:
        return None
    text = str(path).replace("\\", "/")
    return text.rsplit("/", 1)[-1] or None


def _job_view(run_id: str) -> dict[str, Any]:
    job = JOBS.get(run_id) or {}
    run_dir = config.OUTPUT_DIR / run_id
    review = None
    detect = None
    if (run_dir / "review_report.json").is_file():
        try:
            review = json.loads((run_dir / "review_report.json").read_text("utf-8"))
        except Exception:  # noqa: BLE001
            review = None
    if (run_dir / "detect_report.json").is_file():
        try:
            detect = json.loads((run_dir / "detect_report.json").read_text("utf-8"))
        except Exception:  # noqa: BLE001
            detect = None
    artifacts = []
    if run_dir.is_dir():
        for path in sorted(run_dir.rglob("*")):
            if path.is_file() and path.name != "source_path.txt":
                artifacts.append(
                    {
                        "name": str(path.relative_to(run_dir)).replace("\\", "/"),
                        "bytes": path.stat().st_size,
                    }
                )
    status = job.get("status")
    if not status:
        status = "completed" if review else ("running" if run_id in orchestrator.active else "unknown")
    return {
        "run_id": run_id,
        "status": status,
        "created": job.get("created") or (run_dir.stat().st_ctime if run_dir.exists() else None),
        "finished": job.get("finished"),
        "attempts": job.get("attempts") or (review or {}).get("attempts"),
        "passed": (review or {}).get("passed"),
        "score": (review or {}).get("score"),
        "comment": (review or {}).get("comment"),
        "failures": (review or {}).get("failures"),
        "error": job.get("error"),
        "src_name": job.get("src_name")
        or (detect or {}).get("video", {}).get("name")
        or _basename((detect or {}).get("video", {}).get("path")),
        "video": (detect or {}).get("video"),
        "artifacts": artifacts,
        "has_reports": bool(review or detect),
    }


# --------------------------------------------------------------------------- #
# 页面 & 元信息
# --------------------------------------------------------------------------- #
@app.get("/")
async def index() -> FileResponse:
    if not config.INDEX_HTML.is_file():
        raise HTTPException(500, "index.html 缺失")
    return FileResponse(config.INDEX_HTML, media_type="text/html; charset=utf-8")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "app": config.APP_NAME,
        "version": config.APP_VERSION,
        "pid": os.getpid(),
        "port": config.PORT,
        "uptime": round(time.time() - _STARTED_AT, 1),
        "ffmpeg": _ffmpeg_status(),
        "llm": llm_module.get_client().status,
        "runtime": {
            "python": os.sys.version.split()[0],
            "workspace": str(config.WORKSPACE_DIR),
            "outputs": str(config.OUTPUT_DIR),
        },
        "config": config.summary(),
        "active": sorted(orchestrator.active),
    }


@app.get("/api/dag")
async def dag() -> dict[str, Any]:
    data = orchestrator.describe()
    data["llm"] = llm_module.get_client().status
    return data


# --------------------------------------------------------------------------- #
# SSE：先订阅再推，事件按 run_id 过滤
# --------------------------------------------------------------------------- #
@app.get("/api/events")
async def events(request: Request, run_id: str | None = None):
    queue = bus.subscribe()

    async def gen():
        try:
            ready = json.dumps(
                {"type": "stream_ready", "run_id": run_id, "ts": _now()},
                ensure_ascii=False,
            )
            yield f"data: {ready}\n\n"
            # 连接建立后先把该 run 的历史事件补一遍（页面刷新/断线重连不丢上下文）
            if run_id:
                for raw in bus.history(run_id):
                    yield f"data: {raw}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    raw = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                if run_id:
                    try:
                        evt = json.loads(raw)
                    except Exception:  # noqa: BLE001
                        evt = {}
                    if evt.get("run_id") not in (None, run_id):
                        continue
                yield f"data: {raw}\n\n"
        finally:
            bus.unsubscribe(queue)

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)


# --------------------------------------------------------------------------- #
# 上传 / 示例视频
# --------------------------------------------------------------------------- #
@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    name = Path(file.filename or "input.mp4").name
    ext = Path(name).suffix.lower()
    if ext not in config.ALLOWED_EXTS:
        raise HTTPException(400, f"不支持的格式 {ext}，仅支持 {sorted(config.ALLOWED_EXTS)}")

    token = f"up_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    dest_dir = config.UPLOAD_DIR / token
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    try:
        with dest.open("wb") as fh:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"文件超过 {config.MAX_UPLOAD_MB}MB 限制")
                fh.write(chunk)
    except HTTPException:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise
    finally:
        await file.close()

    if size == 0:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise HTTPException(400, "空文件")

    try:
        info = await asyncio.to_thread(wu.probe_video, dest)
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise HTTPException(400, f"视频无法解析：{exc}") from exc

    info["name"] = name
    return {"input_path": str(dest), "name": name, "bytes": size, "video": info}


@app.post("/api/sample")
async def sample(seconds: int = 6, with_audio: bool = True) -> dict[str, Any]:
    seconds = max(3, min(15, int(seconds)))
    token = f"sample_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    dest_dir = config.UPLOAD_DIR / token
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "sample_watermark.mp4"
    try:
        meta = await asyncio.to_thread(
            sample_video.create_sample, dest, seconds, with_audio=with_audio
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"示例视频生成失败：{exc}") from exc
    info = await asyncio.to_thread(wu.probe_video, dest)
    info["name"] = dest.name
    return {
        "input_path": str(dest),
        "name": dest.name,
        "bytes": dest.stat().st_size,
        "video": info,
        "note": meta.get("note", ""),
    }


# --------------------------------------------------------------------------- #
# 启动任务
# --------------------------------------------------------------------------- #
class GenerateRequest(BaseModel):
    run_id: str
    input_path: str
    authorized: bool = False
    params: dict[str, Any] = Field(default_factory=dict)


async def _run_job(run_id: str, payload: dict[str, Any]) -> None:
    job = JOBS.setdefault(run_id, {})
    job["status"] = "running"
    job["started"] = _now()
    try:
        result = await orchestrator.run_job(run_id, payload)
        job["status"] = result.get("status", "completed")
        job["attempts"] = result.get("attempt")
        job["error"] = result.get("error")
        job["review"] = result.get("review")
        job["artifacts"] = result.get("artifacts")
    except asyncio.CancelledError:
        job["status"] = "cancelled"
        raise
    except Exception as exc:  # noqa: BLE001
        job["status"] = "failed"
        job["error"] = f"{type(exc).__name__}: {exc}"
        bus.emit("task_failed", run_id, f"任务异常：{job['error']}", error=job["error"])
    finally:
        job["finished"] = _now()
        TASKS.pop(run_id, None)


@app.post("/api/generate")
async def generate(req: GenerateRequest) -> dict[str, Any]:
    if not req.authorized:
        raise HTTPException(403, "必须确认「仅处理本人拥有版权或已获授权的内容」")
    run_id = req.run_id.strip()
    if not _RUN_ID_RE.match(run_id):
        raise HTTPException(400, "run_id 需为 6-64 位字母/数字/-/_，且以字母或数字开头")
    if run_id in orchestrator.active or run_id in TASKS:
        raise HTTPException(409, f"任务 {run_id} 正在运行")

    src = Path(req.input_path)
    if not src.is_absolute():
        src = (config.WORKSPACE_DIR / src).resolve()
    if not src.is_file():
        raise HTTPException(400, f"找不到输入文件：{src}")
    if not str(src.resolve()).startswith(str(config.WORKSPACE_DIR.resolve())):
        raise HTTPException(403, "输入文件必须位于工作区内")
    if src.suffix.lower() not in config.ALLOWED_EXTS:
        raise HTTPException(400, "不支持的输入格式")

    run_dir = config.OUTPUT_DIR / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)  # 同 id 重跑：清掉旧产物
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "source_path.txt").write_text(str(src), encoding="utf-8")

    params = {
        "sample_frames": req.params.get("sample_frames"),
        "appear_rate": req.params.get("appear_rate"),
        "max_regions": req.params.get("max_regions"),
        "min_conf": req.params.get("min_conf"),
    }
    params = {k: v for k, v in params.items() if v is not None}

    payload: dict[str, Any] = {
        "run_id": run_id,
        "src_path": str(src),
        "run_dir": str(run_dir),
        "stage_dir": str(run_dir / "attempt_1"),
        "params": params,
        "authorized": True,
        "attempt": 1,
        "feedback_count": 0,
        "trace": [],
    }
    JOBS[run_id] = {
        "status": "queued",
        "created": _now(),
        "src_name": src.name,
        "params": params,
    }
    bus.emit(
        "task_queued",
        run_id,
        f"任务已入队：{src.name}",
        src=str(src),
        params=params,
        run_dir=str(run_dir),
    )
    TASKS[run_id] = asyncio.create_task(_run_job(run_id, payload))
    return {"run_id": run_id, "status": "queued", "dag": orchestrator.layers}


# --------------------------------------------------------------------------- #
# 查询 / 取文件
# --------------------------------------------------------------------------- #
@app.get("/api/runs")
async def runs() -> dict[str, Any]:
    items = [_job_view(rid) for rid in _scan_runs()]
    items.sort(key=lambda x: x.get("created") or 0, reverse=True)
    return {"runs": items, "active": sorted(orchestrator.active), "count": len(items)}


@app.get("/api/runs/{run_id}")
async def run_detail(run_id: str) -> dict[str, Any]:
    if not _RUN_ID_RE.match(run_id):
        raise HTTPException(400, "run_id 不合法")
    run_dir = config.OUTPUT_DIR / run_id
    if not run_dir.is_dir():
        raise HTTPException(404, "任务不存在")
    view = _job_view(run_id)
    reports: dict[str, Any] = {}
    for key, fname in (
        ("detect", "detect_report.json"),
        ("review", "review_report.json"),
        ("run_log", "run_log.json"),
    ):
        path = run_dir / fname
        if path.is_file():
            try:
                reports[key] = json.loads(path.read_text("utf-8"))
            except Exception as exc:  # noqa: BLE001
                reports[key] = {"error": str(exc)}
    view["reports"] = reports
    # 前端直接读 review / detect，避免在 JS 里翻 reports 的嵌套结构
    view["review"] = reports.get("review")
    view["detect"] = reports.get("detect")
    view["events"] = bus.history(run_id, 200)
    return view


@app.get("/api/runs/{run_id}/video")
async def run_video(run_id: str) -> FileResponse:
    target = _safe_run_file(run_id, "clean.mp4")
    return FileResponse(target, media_type="video/mp4", filename=f"{run_id}_clean.mp4")


@app.get("/api/runs/{run_id}/source")
async def run_source(run_id: str) -> FileResponse:
    """回放原始输入（不复制文件，靠 run 目录里的 source_path.txt 指路）。"""
    if not _RUN_ID_RE.match(run_id):
        raise HTTPException(400, "run_id 不合法")
    marker = config.OUTPUT_DIR / run_id / "source_path.txt"
    if not marker.is_file():
        raise HTTPException(404, "该任务没有记录输入路径")
    src = Path(marker.read_text("utf-8").strip())
    if not src.is_file():
        raise HTTPException(404, f"原始文件已不在：{src.name}")
    return FileResponse(src, media_type="video/mp4")


@app.get("/api/runs/{run_id}/files/{rel_path:path}")
async def run_file(run_id: str, rel_path: str) -> FileResponse:
    target = _safe_run_file(run_id, rel_path)
    suffix = target.suffix.lower()
    media = {
        ".mp4": "video/mp4",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".json": "application/json",
        ".txt": "text/plain; charset=utf-8",
        ".md": "text/markdown; charset=utf-8",
    }.get(suffix, "application/octet-stream")
    return FileResponse(target, media_type=media)


@app.get("/favicon.ico")
async def favicon() -> JSONResponse:
    return JSONResponse({}, status_code=204)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")