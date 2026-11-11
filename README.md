# 视频自动去水印工具

一个**本地运行的多智能体（Multi-Agent）Web 服务**：上传一段带固定水印（台标 / 角标 / LOGO / 半透明文字）的视频，流水线自动完成
「检测水印 → 生成遮罩 → 修复 → 精修 → 质检」，质检不达标会自动回灌参数重跑，最终输出 `clean.mp4` 与两份可追溯的报告。

核心目标不是「把水印糊掉」，而是**修复后肉眼与指标都看不出修痕**：修复区的纹理方差、边缘密度、亮度都必须与紧邻的背景接续。
所有视频处理在本地用 OpenCV + ffmpeg 完成，只有 3 个「做决策」的小环节可选调用 LLM，且没有 API Key 时自动降级为规则化决策，功能完全可用。

> 🌐 **在线展示页**：<https://htedot.github.io/video-dewatermark/>
> Pages 只能托管静态文件，所以那一页是「一次真实运行的产物回放」——视频、日志、评分、截图都来自本服务的真实 run；
> 真正的处理仍需在本机运行本项目。源码同步在 <https://github.com/htedot/video-dewatermark/tree/main/source>。

> ⚠️ **授权声明（必读）**
> 本工具**仅**用于处理你本人拥有版权、或已取得权利人明确授权的视频内容。
> **禁止**用它去除他人作品的水印、台标、版权标识或任何用于规避版权保护的内容。
> 页面上的「开始」按钮必须先勾选授权确认才会启用，后端 `POST /api/generate` 也会校验 `authorized=true`，否则返回 403。

---

## 一、项目作用

### 1.1 它能做什么

| 能力 | 说明 |
| --- | --- |
| 固定位置水印检测 | 多帧采样 + 「中位帧边缘 ∩ 多帧出现率」双重投票，只保留在绝大多数帧里都出现、且位置稳定的区域 |
| 半透明叠加层识别 | 半透明水印会整体压低该区域的时间方差，检测到「衰减」后按**整框**生成遮罩，而不是只盖笔画（否则会留下淡影） |
| 遮罩羽化 | 二值遮罩经 `GaussianBlur(mask, (0,0), sigma)` 羽化（sigma 5–10），边界软合成，避免硬边 |
| 修复 | `cv2.inpaint`（Telea / NS，radius 3–8）；修复质量不足时启用本项目的增量：**邻近真实块移植（patch fill）+ 高频纹理回填** |
| 质量质检 | 抽 5 帧，逐区域算「高频纹理方差比 / 边缘密度比 / 亮度差」，四维加权打分（阈值 75 分 + 逐项硬阈值） |
| 不达标回灌 | 质检失败 → 生成下一轮参数（method / radius / blur_mix / 羽化 / 膨胀 / ink 阈值 / patch 开关）→ 回到遮罩阶段重跑，最多 2 轮 |
| 可追溯产物 | `clean.mp4`、`detect_report.json`、`review_report.json`、`run_log.json`、检测/遮罩/质检截图 |
| 实时可视化 | SSE 推送每个 Agent 的开始/进度/完成、回灌事件与最终结果，前端绘制 DAG 节点状态与日志 |

### 1.2 架构

```
浏览器（index.html，单文件无 CDN）
   │  EventSource /api/events?run_id=...   ← 先订阅
   │  POST /api/generate                  ← 再提交任务
   ▼
FastAPI（backend/main.py）
   │
Orchestrator（唯一持有全部 Agent 的地方）
   │
   ├─ detector   检测：抽帧 → 静态图 → 候选框（bbox + 置信度）
   ├─ mask       遮罩：候选 → 半透明判定 → 笔画/整框 → 羽化
   ├─ inpainter  修复：参数选择 → cv2.inpaint →（回灌轮）patch 移植 + 纹理回填
   ├─ refiner    精修：对比修复区与邻域，超阈值按问题类型重修
   ├─ reviewer   质检：抽 5 帧打分 → 通过 / 不通过 + 下一轮参数
   ├─ preview    预览：原片（带检测框）vs 成品 并排图
   └─ report     报告：detect_report.json / review_report.json / run_log.json
        │
        └─ reviewer 不达标 ──review_feedback──▶ 回到 mask（最多 2 次）
```

- **只有 `Orchestrator` 允许 import 各个业务 Agent**，Agent 之间零 import，全部通过 `payload` 字典传数据（`payload["detect"]` → `payload["mask"]` → …）。
- 每个 Agent 只实现 `async def run(payload) -> payload`，业务异常抛 `AgentError`，编排器统一 `emit agent_failed` 并终止任务。
- 事件总线 `EventBus` 是同步 API：`subscribe()` 返回 `asyncio.Queue`，`publish()` 内部 `json.dumps` 后 fanout，SSE 生成器只做 `yield f"data: {raw}\n\n"`；历史事件按 `run_id` 隔离回放。

### 1.3 本项目的增量（任务书之外的自己加的东西）

1. **检测种子改造**（`watermark_utils._seed_ink`）：把「出现率 > 0.7 的投票图」直接当连通域输入会得到上百个运动残影噪声块；改成
   `ink = Canny(GaussianBlur(中位帧)) ∩ (support >= 出现率阈值)`，在自造样本上候选从 **133 个降到 2 个**，且两个都是真水印。
   出现率的语义仍然是任务书要求的「多帧投票出现率」，只是种子改由中位帧边缘提供。
2. **邻近真实块移植 + 高频纹理回填**（`watermark_utils._patch_fill` / `_texture_restore`）：`cv2.inpaint` 本质是扩散，修完必然是一块「磨皮斑」。
   本项目在回灌轮额外做两件事：从紧邻的背景里搬一块真实纹理贴过来（按外圈均值/标准差选最优源块，做均值平移后按羽化 alpha 软合成），
   再把缺失的高频能量按 `var_target = var_base + s²·var_field` 解出增益补回去。样本上质检分从 **46.55 → 80.9**，肉眼无修痕。
3. **质检判据换成高频纹理方差比**（`tex_ratio`）：整块方差被背景大尺度渐变主导，`|var_in−var_ring|/var_ring` 在原片上就能到 2–10，不可用；
   改用 `hp = gray − GaussianBlur(gray, σ=8)` 的方差比，同一块背景在修复前后可比。原 `var_ratio` 保留为诊断值。
4. **中位数 + 帧通过率聚合**：逐帧指标会被「路过的运动物体」污染，5 帧逐帧都必须过太苛刻；改为「5 帧中位数判阈值 + 至少 60% 的帧单帧达标」，
   并在环带本身几乎没有边缘（`edge_ring < 0.02`）时豁免边缘密度判据——低纹理背景上「边缘密度比 ≥ 0.35」物理上不成立。
5. **回灌参数梯度表**（`data/inpaint_grid.json`）：第 1 轮是任务书 baseline（纯 `cv2.inpaint`，快），第 2/3 轮逐级升到 patch 移植 + 更强羽化/膨胀，
   `region_overrides` 再按「边缘残留 / 过度平滑」做**逐区域**微调。多轮产物落 `attempt_1/`、`attempt_2/`，互不覆盖。

---

## 二、部署方法

### 2.1 环境要求

- Windows 10/11 x64（脚本是 .bat；Linux/macOS 可直接用 `python backend/main.py`）
- 便携 Python 3.11+（本项目自带 `runtime/python`，**不创建 venv**）
- ffmpeg + ffprobe（本项目自带 `runtime/ffmpeg`，需含 libx264；ffprobe 用于探测音轨/帧数）

### 2.2 目录约定

```
<workspace>/
├─ runtime/
│  ├─ python/python.exe        # 便携 Python（已装好依赖）
│  └─ ffmpeg/{ffmpeg,ffprobe}.exe
└─ video_dewatermark/          # 本项目（任务书里的 my_project/）
   ├─ start.bat  stop.bat  run_tests.bat  README.md
   ├─ tests/
   └─ backend/
      ├─ main.py  config.py  .env  llm.py  event_bus.py  watermark_utils.py
      ├─ sample_video.py  index.html
      ├─ agents/{base,orchestrator,detector_agent,mask_agent,inpainter_agent,
      │          refiner_agent,reviewer_agent,preview_agent,report_agent}.py
      ├─ data/{watermark_presets,inpaint_grid,review_rules}.json + prompts/
      └─ outputs/<run_id>/...
```

`backend/config.py` 会按 `runtime/ffmpeg → 项目 tools/ → PATH` 的顺序找二进制，找不到会在 `/api/health` 里报 `ffmpeg.ok=false`。

### 2.3 安装依赖（一次性）

```powershell
# 依赖装进项目自带的 portable python
E:\123\runtime\python\python.exe -m ensurepip --upgrade
E:\123\runtime\python\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple `
  "fastapi>=0.115" "uvicorn[standard]>=0.30" "opencv-python>=4.11" "numpy>=2.1" `
  "pillow>=11" "openai>=1.60" "python-dotenv>=1.0" "pydantic>=2.9" "httpx>=0.27" `
  "python-multipart>=0.0.9" "imageio-ffmpeg>=0.5"
```

> `python-multipart` 必须装：`POST /api/upload` 用的是 `UploadFile`，缺它 FastAPI 在**导入期**就会抛
> `Form data requires "python-multipart" to be installed`，整个服务起不来。
> `imageio-ffmpeg` 只是用来取一份带 libx264 的 Windows x64 ffmpeg（本项目已把它拷到 `runtime/ffmpeg`，可不装）。

### 2.4 配置（`backend/.env`）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `PORT` | `8231` | 服务端口，必须在 8200–8299；`start.bat` 会先用 netstat 检查占用 |
| `HOST` | `127.0.0.1` | 只监听本机 |
| `JOB_TIMEOUT` | `1800` | 单条任务超时（秒），超时 emit `task_failed` |
| `DEEPSEEK_API_KEY` | 空 | **留空也能全功能运行**（LLM 环节走规则降级） |
| `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL` | `https://api.deepseek.com` / `deepseek-chat` | OpenAI 兼容接口 |
| `LLM_TIMEOUT` / `LLM_MAX_TOKENS` / `LLM_RETRIES` | `90` / `1400` / `2` | 超时 + 重试；失败即降级，绝不中断流水线 |
| `DETECT_SAMPLE_FRAMES` | `48` | 检测采样帧数（前端可覆盖 16–120） |
| `DETECT_WORK_WIDTH` | `960` | 检测前把画面缩到的宽度（提速） |
| `DETECT_APPEAR_RATE` | `0.70` | 候选框出现率阈值（前端可覆盖 0.30–0.95） |
| `DETECT_MIN_CONF` | `0.45` | 进入复核的最低置信度 |
| `MASK_FEATHER_SIGMA` | `7.0` | 羽化高斯 sigma（5–10） |
| `INPAINT_RADIUS` | `5` | `cv2.inpaint` 半径（3–8） |
| `QUALITY_VAR_RATIO` / `QUALITY_EDGE_RATIO` | `0.30` / `0.85` | 质检阈值（`review_rules.json` 为准，这里是默认值） |
| `REVIEW_FRAMES` | `5` | 质检抽帧数 |
| `MAX_FEEDBACK` | `2` | 回灌最大轮数 |
| `VIDEO_CRF` / `VIDEO_PRESET` | `18` / `veryfast` | H.264 编码质量/速度 |
| `MAX_UPLOAD_MB` | `512` | 上传大小上限 |
| `WM_OUTPUT_DIR` / `WM_RUNTIME_DIR` | `backend/outputs` / `<workspace>/runtime` | 产物与运行时目录 |

改 `.env` 或任何 `.py` 之后必须**重启服务**（`stop.bat` → `start.bat`）。本项目刻意不加 `--reload`：视频处理是长任务，热重载会把正在跑的 job 杀掉。

---

## 三、运行方式

### 3.1 双击脚本（推荐）

```
start.bat   # 检查 runtime → 检查端口 → 后台起服务 → 轮询 /api/health → 打开浏览器
stop.bat    # 读 backend/logs/server.pid → taskkill /T /F → 兜底按端口杀
```

- 服务在后台隐藏窗口运行，日志：`backend/logs/server.log`（stdout）、`backend/logs/server.err.log`（stderr）、PID：`backend/logs/server.pid`。
- 两个 .bat 都是**纯 ASCII + CRLF + 无 BOM**（任务书硬要求，cmd 对 BOM/裸 LF 会莫名报错），契约测试里有对应断言。

### 3.2 手工启动

```powershell
cd E:\123\video_dewatermark
E:\123\runtime\python\python.exe -u -X utf8 backend\main.py
# 浏览器打开 http://127.0.0.1:8231/
```

### 3.3 跑测试

```
run_tests.bat                                  # 或
E:\123\runtime\python\python.exe tests\run_tests.py
```

- `tests/test_contracts.py`（17 项，秒级）：`subscribe()` 是同步方法、`publish()` 返回 JSON 字符串、SSE 格式、事件按 run_id 隔离、
  `_extract_json` 对 fenced / 嵌套花括号 / 截断文本的处理、`complete_json` 只认 dict、`complete_json_list` 认 list、
  `complete_text` 返回 `(text, tokens)`、无 Key 时 `call_llm` 降级不抛、Agent 之间零 import、编排器 6 层 + 回灌边指向 mask、
  端口在 8200–8299、ffmpeg/ffprobe 可执行、.bat 三件套 ASCII+CRLF+无 BOM、候选检测 / 遮罩羽化 / 质检指标正确性。
- `tests/test_pipeline.py`（2 项，1–2 分钟）：**完整链路测试**——用 `sample_video.create_sample` 造带音轨带水印的样本 →
  `Orchestrator.run_job` 跑完 → 断言 `clean.mp4` 可探测、分辨率一致、音轨保留、`passed=true`、5 张质检图 + montage 落盘、
  报告内容一致；发生过回灌时断言第 2 轮确实启用了 patch 移植/纹理回填，且日志里能查到 `review_feedback`。

---

## 四、使用方法

### 4.1 页面操作（`http://127.0.0.1:8231/`）

1. 顶部三颗状态灯：服务 / ffmpeg / LLM（无 Key 时显示「未配置（规则降级）」，属正常）。
2. **勾选授权声明确认框**——不勾选「开始」按钮是禁用的。
3. 选输入（三种方式）：
   - **本地上传**：拖拽或点选视频（mp4/mov/mkv/avi/webm，≤512MB）；
   - **服务器路径**：直接填绝对路径（必须在工作区目录内）；
   - **内置示例**：一键生成 6 秒带音轨、带两处固定水印的测试视频（`POST /api/sample`）。
4. 调参数（一般不用动）：采样帧数 / 出现率阈值 / 最多区域数 / 最低置信度。
5. 点「开始去水印」：前端**先**建 `EventSource`、等 `readyState===1` 才发 `POST /api/generate`，避免漏掉开头的事件。
6. 看实时日志与 DAG 节点；质检不达标时会出现橙色 `review_feedback` 行，`mask` → `reviewer` 节点回到 running，`attempt` +1。
7. 完成后结果区给出：分数/verdict/轮数、四维得分、失败原因（若未通过）、原片与成品双播放器、质检截图墙、产物下载表。

### 4.2 API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 单文件前端 |
| GET | `/api/health` | 服务/ffmpeg/LLM/端口/当前活跃任务 |
| GET | `/api/dag` | 流水线分层、节点依赖、回灌边 |
| GET | `/api/events?run_id=` | SSE 事件流（先发 `stream_ready`，再回放该 run 的历史事件，15s keep-alive） |
| POST | `/api/upload` | `multipart/form-data`，字段名 `file` → `{input_path, video}` |
| POST | `/api/sample?seconds=6` | 生成内置示例视频 → `{input_path, ...}` |
| POST | `/api/generate` | `{run_id, input_path, authorized:true, params:{sample_frames,appear_rate,max_regions,min_conf}}` |
| GET | `/api/runs` | 历史任务列表 |
| GET | `/api/runs/{run_id}` | 合并报告 + 事件历史 + 产物列表 |
| GET | `/api/runs/{run_id}/video` | `clean.mp4`（支持 Range，可拖动播放） |
| GET | `/api/runs/{run_id}/source` | 原片（从 `source_path.txt` 读取，不复制大文件） |
| GET | `/api/runs/{run_id}/files/{path}` | 任意产物（路径做越界校验，只允许 run 目录内） |

```bash
# 1) 上传
curl -F "file=@demo.mp4" http://127.0.0.1:8231/api/upload
# 2) 提交任务（run_id 必须匹配 ^[A-Za-z0-9][A-Za-z0-9_-]{5,63}$）
curl -X POST http://127.0.0.1:8231/api/generate -H "Content-Type: application/json" ^
  -d "{\"run_id\":\"demo001\",\"input_path\":\"<上一步的 input_path>\",\"authorized\":true,\"params\":{\"sample_frames\":48,\"appear_rate\":0.7}}"
# 3) 看事件（另开窗口）
curl -N "http://127.0.0.1:8231/api/events?run_id=demo001"
# 4) 取结果
curl -o clean.mp4 http://127.0.0.1:8231/api/runs/demo001/video
```

### 4.3 产物说明（`backend/outputs/<run_id>/`）

| 文件 | 内容 |
| --- | --- |
| `clean.mp4` | 成品（H.264 + 原音轨，`+faststart`） |
| `detect_report.json` | 采样信息（帧号/工作分辨率/缩放比/出现率阈值）、全部候选（bbox、出现率、稳定性、边缘强度、位置先验、置信度、zone）、选中区域、复核来源（rules/llm + 降级标记） |
| `review_report.json` | 是否通过、分数/通过线、轮数、四维得分、逐区域指标（tex_ratio、edge_ratio、lum_diff、pass_frames）、失败原因、下一轮参数、人工复核图清单 |
| `run_log.json` | 全流程事件日志（与 SSE 同源） |
| `detect_preview.jpg` / `detect_support.jpg` | 检测框叠加图 / 多帧出现率热力图 |
| `mask.png` / `mask_heat.png` | 二值遮罩 / 羽化热力图 |
| `review_frames/frame_01..05.jpg` | 每个质检点「原片切片 \| 修复后切片」并排图（**人工复核就看这 5 张**） |
| `review_frames/montage.jpg` | 5 帧拼图 |
| `preview_before_after.jpg` | 3 组原片（带框）/ 成品并排预览 |
| `attempt_N/{inpainted,clean}.mp4` | 每一轮的中间产物，便于对比「第 1 轮 vs 第 2 轮」修得如何 |

### 4.4 一次真实运行的样子（内置示例，1280×720 / 30fps / 4s）

```
detector  48 帧采样，工作区 960px → 候选 2 个
          c1 bbox=(690,25,236,20) appear=0.998 conf=0.874 zone=top_corner
          c2 bbox=(26,467,140,47) appear=0.999 conf=0.899 zone=bottom_corner
mask      覆盖 3.19%，半透明判定 attenuated=true（整框），羽化 σ=7
inpainter telea / r5 / blur_mix=0（baseline）
refiner   ns / r8 重修 2 个区域
reviewer  46.55 分 → 不达标：纹理不接续（c1 0.78 / c2 0.85 > 0.3）
          → review_feedback：回灌第 2 轮
mask      覆盖 3.63%，σ=9
inpainter ns / r8 + patch_fill + texture_restore
reviewer  80.9 分 → 通过（c1 tex 0.124 / c2 0.039，边缘环带无纹理已豁免，4/5 帧达标）
report    「共 2 轮，产物 19 个」
```

---

## 五、未来扩展

1. **动态/滚动水印**：目前假设水印位置固定。可按帧做「运动补偿后的时间中值」，或用轻量分割模型（YOLO-seg / SAM2）逐帧出 mask，再交给现有 inpainter/质检。
2. **更强修复后端**：`inpaint_video` 已把「方法」抽象成 `telea/ns/blur`，可插 LaMa（`iopaint`）或视频扩散模型；质检与回灌逻辑无需改动。
3. **光流引导补全**：patch 移植目前是「同帧邻域取块」。跨帧取块（同区域在前后的真实背景）能解决大面积遮挡，也能让镜头移动时更稳。
4. **多水印/多尺度**：现在工作宽度 960、一次检测。可对 4K 源做金字塔多尺度检测，并按区域分别给出参数。
5. **质量判据升级**：接 LPIPS / CLIP-IQA 与 `tex_ratio` 融合，减少「指标过了但人眼仍觉得糊」的情况；把人工复核结果回写成标注，做阈值自校准。
6. **批处理与队列**：`JOBS/TASKS` 现在是进程内字典，可换 Redis + 任务队列，支持多机 worker 与断点续跑。
7. **LLM 能力扩展**：目前 3 个触点（候选复核、修复参数、质检评语）。可加「失败原因归因」与「参数搜索策略」，但必须保持「LLM 只做决策、不做像素」与「失败即规则降级」两条底线。

---

## 六、踩坑记录

> 以下都是本项目真实踩到并已修掉的坑，按「现象 → 原因 → 处理」记录，方便下次直接绕开。

1. **ffmpeg 报「不是此操作系统平台的有效应用程序」**
   现象：`E:\ffmpeg\...winarm64...\bin\ffmpeg.exe` 直接运行失败。
   原因：拿到了 **win-arm64** 构建，本机是 x64。
   处理：改用 `imageio-ffmpeg` 车轮里自带的 `ffmpeg-win-x86_64-v9.0.2.exe`（gyan essentials，含 libx264 + aac），拷到 `runtime/ffmpeg/ffmpeg.exe`；
   ffprobe 用本机已有的 4.3.2。**注意**：机器上另一份 ffmpeg 4.3.2（GPT-SoVITS 自带）**没有 libx264**，不能用来编码 H.264。

2. **便携 Python 拷过来没有 pip**
   现象：`runtime\python\python.exe -m pip` 报 `No module named pip`。
   原因：rcopy 时排除了 `site-packages`，`ensurepip` 的 wheels 也在里面。
   处理：`python -m ensurepip --upgrade` 重新注入 pip，再装依赖。

3. **默认 PyPI 装依赖卡死、gyan.dev 只有 ~25KB/s**
   现象：`pip install` 长时间无输出直到超时。
   处理：换清华镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple`，1 分钟装完。
   （pip 的 stderr 警告会让 PowerShell 任务退出码变 1，**不代表安装失败**，要实际 import 验证。）

4. **服务起不来：`Form data requires "python-multipart" to be installed`**
   原因：`UploadFile` 路由需要 `python-multipart`，缺它在 **import 期**就抛，不是运行时。
   处理：装 `python-multipart`。这类「导入期就炸」的依赖建议写进部署清单并加一条 health 自检。

5. **`subscribe()` 写成 async 导致 SSE 拿到协程对象**
   原因：`bus.subscribe()` 被 `await` 后返回的是 `Queue` 的协程结果，生成器里再 `queue.get()` 就报错；更隐蔽的是「看起来能连上但没有事件」。
   处理：`subscribe()` 保持同步，内部只做 `asyncio.Queue()` + 注册；契约测试 `test_bus_subscribe_is_sync` 固定住这一点。

6. **SSE 前几条事件丢失**
   原因：前端先 `POST /api/generate` 再建 `EventSource`，`task_queued`/`agent_started` 已经发完了。
   处理：前端严格「先 `EventSource` → 等 `readyState===1` → 再 POST」；服务端在 `stream_ready` 之后回放 `bus.history(run_id)`，双保险。

7. **LLM 返回的 JSON 直接 `json.loads` 崩**
   原因：模型常包 ```json 代码块、前后加解释、或字符串里带花括号。
   处理：`_extract_json` 做**平衡括号扫描**（跳过字符串与转义），只在扫描出完整结构时解析；失败返回 `None` 走规则降级。
   `complete_json` 顶层只认 dict，要 list 用 `complete_json_list`——这条也进了契约测试。

8. **候选框 133 个，全是运动残影**
   原因：把「出现率 > 0.7 的投票图」直接做闭运算，噪声被连成大片；示例视频里静态棋盘格背景是元凶之一。
   处理：种子改为「中位帧 Canny ∩ 投票图」，并把示例视频底图改成**每帧重画的平滑噪声 + 漂移物体**（不再有静态棋盘格）。候选降到 2 个。

9. **`_position_prior` 永远返回 `top_corner`**
   原因：调用处传的是绝对坐标，函数内部又除了一次宽高，等于归一化了两次。
   处理：签名改成 `_position_prior(nx, ny, presets)`，只接受 0–1 归一化输入。

10. **`UnboundLocalError: contrast_norm`**
    原因：闭包内又定义了同名局部变量，Python 把它当局部作用域，外层参数被遮蔽。
    处理：外层参数改名 `tstd_scale/grad_scale/contrast_scale`；这类「闭包同名变量」在重构时很容易漏。

11. **`BaseAgent.emit() got multiple values for keyword argument 'attempt'`**
    原因：`completed()` 已经自动带 `attempt`，调用处又显式传了一次。
    处理：业务 Agent 不再手传 `attempt`；事件字段统一由 `BaseAgent.emit/started/progress/completed` 组装。

12. **修复完是一块「磨皮斑」，右上文字还有淡白残影**
    原因：① `cv2.inpaint` 只做扩散，纹理能量全丢；② 半透明水印是整块叠加，只盖笔画会留影。
    处理：① 回灌轮启用 patch 移植 + 高频纹理回填；② `mask` 里用「框内 tstd / 环带 tstd」判定半透明（样本上 c1 0.59 / c2 0.47），
    低于阈值按整框填充并强制 `dilate_extra ≥ 2`。

13. **质检分数与肉眼完全相反**
    原因：`|var_in − var_ring| / var_ring` 用的是整块方差，被背景大尺度渐变主导；原片自己的比值就有 2–10。
    处理：改用高频纹理方差比 `tex_ratio`（`hp = gray − GaussianBlur(gray, σ=8)`），同一块背景修复前后才可比；
    逐帧指标受运动物体干扰，改为 5 帧中位数 + 帧通过率；低纹理环带（`edge_ring < 0.02`）豁免边缘密度判据。

14. **`outputs/offline02/` 只有 `attempt_1/`，参数却已是第 3 档**
    原因：`stage_dir` 只在 `run_job` 里设一次，回灌后没切目录，多轮产物互相覆盖。
    处理：`_check_feedback` 里按 `attempt` 新建 `attempt_N/` 并写回 `payload["stage_dir"]`。

15. **`KeyError: 'run_dir'`（自测脚本直接调 `run_job`）**
    原因：编排器假定 HTTP 层已经塞了 `run_dir`。
    处理：`run_job` 里 `payload.setdefault("run_dir", ...)`，让编排器可独立使用（测试就靠这条直接跑全链路）。

16. **PowerShell 里 `Invoke-RestMethod` 看到的中文是乱码**
    原因：控制台编码，不是服务端。处理：用 Python 客户端或 `curl` 验证；`server.log` 用 UTF-8 写。

17. **.bat 里出现中文/裸 LF/BOM**
    现象：cmd 报「不是内部或外部命令」或括号块解析错乱。
    处理：.bat 一律纯 ASCII + CRLF + 无 BOM，用 Python `bytes` 直写，契约测试 `test_bat_files_ascii_crlf_no_bom` 守住。

---

## 七、附：测试与验收清单

| 验收项 | 落点 | 状态 |
| --- | --- | --- |
| 5 个 Agent 各司其职、只实现 `run(payload)` | `backend/agents/*.py` | ✅ |
| Agent 间零 import、数据只走 payload | `test_agents_do_not_import_each_other` | ✅ |
| 多帧投票出现率 > 0.7 | `watermark_utils.generate_candidates`、`detect_report.json` | ✅ |
| 羽化 mask（高斯 5–10px） | `build_masks` + `mask_heat.png` | ✅ |
| `cv2.inpaint` Telea/NS + radius 3–8 | `inpaint_video`、`repair.params` | ✅ |
| 修复区与邻域方差对比 + 超阈值重修 | `refiner_agent`、`reviewer_agent` | ✅ |
| 抽 5 帧人工评分点 | `review_frames/frame_01..05.jpg` | ✅ |
| 不达标调参重修（回灌 ≤2 次） | `review_feedback` 事件 + `attempt_N/` | ✅ |
| 授权声明 + 后端 403 校验 | `index.html`、`POST /api/generate` | ✅ |
| 产物 `clean.mp4` + `detect_report.json` | `backend/outputs/<run_id>/` | ✅ |
| 完整链路测试（输入→落盘→质检通过） | `tests/test_pipeline.py` | ✅ |
| 日志可见「不达标 → 重生成 → 达标」 | SSE 事件序列 / `run_log.json` | ✅ |

---

## 八、本仓库的目录结构

这个仓库同时是展示页与源码仓库：

| 路径 | 说明 |
| --- | --- |
| `index.html` | 展示页（静态，无 CDN），即 <https://htedot.github.io/video-dewatermark/> |
| `assets/` | 回放的视频、截图、报告与 `run.json`（真实 run `pagedemo01` 的事件流） |
| `source/` | 上面这份项目的完整源码，把内容平铺到 `video_dewatermark/` 后，与 `runtime/` 同级即可按本文档启动 |
