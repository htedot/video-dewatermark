"""全局配置：代码内的默认值 + backend/.env 覆盖。

约定：
* 所有路径都以 ``backend/`` 为锚点，跟着项目目录走，换盘符不用改代码；
* 运行时（Python / ffmpeg）只认工作区 ``runtime/`` 下的受管副本，不联网找东西；
* 这里只放"常量"，任何业务规则都在 ``backend/data/*.json`` 里，方便改规则不改代码。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# 路径锚点
# --------------------------------------------------------------------------- #
BACKEND_DIR = Path(__file__).resolve().parent            # .../backend
PROJECT_DIR = BACKEND_DIR.parent                          # .../video_dewatermark
WORKSPACE_DIR = PROJECT_DIR.parent                        # 工作区根目录

# .env 只做覆盖，不覆盖进程里已经存在的环境变量（start.bat 注入的优先）
load_dotenv(BACKEND_DIR / ".env", override=False)


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or not str(value).strip():
        return default
    return str(value).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(_env(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name, "1" if default else "0").lower()
    return raw in {"1", "true", "yes", "on", "y"}


# --------------------------------------------------------------------------- #
# 服务
# --------------------------------------------------------------------------- #
APP_NAME = _env("APP_NAME", "视频自动去水印工具")
APP_VERSION = "1.0.0"
HOST = _env("HOST", "127.0.0.1")
PORT = _env_int("PORT", 8231)          # 8200-8299 区间内，netstat 确认未被占用

# 单次任务硬超时（秒），防止坏视频把 worker 卡死
JOB_TIMEOUT = _env_int("JOB_TIMEOUT", 1800)

# --------------------------------------------------------------------------- #
# LLM（DeepSeek 文本接口；兼容任何 OpenAI 协议的端点）
# --------------------------------------------------------------------------- #
DEEPSEEK_API_KEY = _env("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = _env("DEEPSEEK_MODEL", "deepseek-chat")
LLM_TIMEOUT = _env_float("LLM_TIMEOUT", 90.0)
LLM_MAX_TOKENS = _env_int("LLM_MAX_TOKENS", 1400)
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.2)
LLM_RETRIES = _env_int("LLM_RETRIES", 2)
# 没配 Key 时是否仍然允许流水线跑（规则化降级）。默认允许，断网/无 Key 不崩。
LLM_ALLOW_OFFLINE = _env_bool("LLM_ALLOW_OFFLINE", True)

# --------------------------------------------------------------------------- #
# 目录
# --------------------------------------------------------------------------- #
DATA_DIR = BACKEND_DIR / "data"
PROMPT_DIR = DATA_DIR / "prompts"
OUTPUT_DIR = Path(_env("WM_OUTPUT_DIR", str(BACKEND_DIR / "outputs")))
UPLOAD_DIR = OUTPUT_DIR / "_uploads"
INDEX_HTML = BACKEND_DIR / "index.html"

for _d in (OUTPUT_DIR, UPLOAD_DIR, PROMPT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# 本地运行时（ffmpeg / ffprobe）：只认 runtime/，其次 PATH
# --------------------------------------------------------------------------- #
RUNTIME_DIR = Path(_env("WM_RUNTIME_DIR", str(WORKSPACE_DIR / "runtime")))


def _find_binary(name: str) -> str | None:
    """按 runtime → 项目 bin → PATH 的顺序找一个可执行文件。"""
    suffix = ".exe" if os.name == "nt" else ""
    candidates = [
        RUNTIME_DIR / "ffmpeg" / f"{name}{suffix}",
        PROJECT_DIR / "tools" / f"{name}{suffix}",
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    found = shutil.which(name)
    return found or None


FFMPEG = _find_binary("ffmpeg")
FFPROBE = _find_binary("ffprobe")

# --------------------------------------------------------------------------- #
# 检测 / 修复算法参数（data/*.json 里还有可调的一份，这里是兜底默认值）
# --------------------------------------------------------------------------- #
DETECT_SAMPLE_FRAMES = _env_int("DETECT_SAMPLE_FRAMES", 48)   # 抽帧数（多帧投票样本）
DETECT_WORK_WIDTH = _env_int("DETECT_WORK_WIDTH", 960)        # 检测时缩放到的宽度
DETECT_APPEAR_RATE = _env_float("DETECT_APPEAR_RATE", 0.70)   # 候选框出现率阈值
DETECT_MIN_CONF = _env_float("DETECT_MIN_CONF", 0.45)         # 进入 LLM/规则复核的最低置信度
MASK_FEATHER_SIGMA = _env_float("MASK_FEATHER_SIGMA", 7.0)    # 羽化高斯 sigma（5-10px）
# 半透明叠加层判定：框内/环带时间方差比低于该值 → 整框修复（防"淡影+涂抹"）
MASK_ATTENUATION_RATIO = _env_float("MASK_ATTENUATION_RATIO", 0.85)
INPAINT_RADIUS = _env_int("INPAINT_RADIUS", 5)                # cv2.inpaint radius（3-8）
QUALITY_VAR_RATIO = _env_float("QUALITY_VAR_RATIO", 0.30)     # |修复区方差-邻域方差|/邻域方差
QUALITY_EDGE_RATIO = _env_float("QUALITY_EDGE_RATIO", 0.85)   # 残留静态边缘 / 邻域边缘密度
REVIEW_FRAMES = _env_int("REVIEW_FRAMES", 5)                  # 质检抽帧数
MAX_FEEDBACK = _env_int("MAX_FEEDBACK", 2)                    # 回灌最大重试次数

# 视频编码
VIDEO_CRF = _env_int("VIDEO_CRF", 18)
VIDEO_PRESET = _env("VIDEO_PRESET", "veryfast")

# 允许的视频后缀
ALLOWED_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}

# 上传大小上限（字节）
MAX_UPLOAD_MB = _env_int("MAX_UPLOAD_MB", 512)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# run_id 白名单（前端生成，服务端当目录名用，必须严格校验）
RUN_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{5,63}$"


def summary() -> dict:
    """给 /api/health 用的配置摘要（不含密钥）。"""
    return {
        "app": APP_NAME,
        "version": APP_VERSION,
        "port": PORT,
        "llm": {
            "configured": bool(DEEPSEEK_API_KEY),
            "base_url": DEEPSEEK_BASE_URL,
            "model": DEEPSEEK_MODEL,
            "timeout": LLM_TIMEOUT,
            "offline_fallback": LLM_ALLOW_OFFLINE,
        },
        "ffmpeg": FFMPEG,
        "ffprobe": FFPROBE,
        "runtime_dir": str(RUNTIME_DIR),
        "output_dir": str(OUTPUT_DIR),
        "detect": {
            "sample_frames": DETECT_SAMPLE_FRAMES,
            "work_width": DETECT_WORK_WIDTH,
            "appear_rate": DETECT_APPEAR_RATE,
            "min_conf": DETECT_MIN_CONF,
        },
        "repair": {
            "feather_sigma": MASK_FEATHER_SIGMA,
            "attenuation_ratio": MASK_ATTENUATION_RATIO,
            "inpaint_radius": INPAINT_RADIUS,
            "var_ratio": QUALITY_VAR_RATIO,
            "edge_ratio": QUALITY_EDGE_RATIO,
            "max_feedback": MAX_FEEDBACK,
        },
    }