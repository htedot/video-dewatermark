# 视频自动去水印工具 · 多智能体流水线

> 检测 → 遮罩 → 修复 → 精修 → 质检：七个 Agent 串成一条可回灌的流水线。
> 全部在本机用 OpenCV + ffmpeg 完成，视频不上传。

**在线展示页（真实运行回放）**：<https://htedot.github.io/video-dewatermark/>

页面里回放的视频、日志、评分、产物都来自本地服务的一次真实运行（run `pagedemo01`，129.5 秒，
第 1 轮 47.12 不达标 → 回灌 → 第 2 轮 76.7 通过）。GitHub Pages 只能托管静态文件，
所以真正的处理需要在你自己的机器上运行下面的后端。

## 目录

| 路径 | 说明 |
| --- | --- |
| `index.html` | 展示页（单文件，无 CDN） |
| `assets/` | 回放数据 `run.json`、压缩后的原片/成品、截图与报告 |
| `source/` | 项目源码（后端、Agent、规则库、测试、启动脚本） |

## 在本地跑起来

`source/` 是完整项目，需要 Python 3.13 + OpenCV + ffmpeg：

```bat
cd source
start.bat        :: 检查端口 8231 → 后台起服务 → 探活 → 打开浏览器
stop.bat         :: 停止
run_tests.bat    :: 19 项测试（17 契约 + 2 端到端）
```

项目默认从上层目录的 `runtime/` 读取解释器与 ffmpeg：

```
video_dewatermark/     <- source/ 放这里
runtime/
  python/python.exe
  ffmpeg/ffmpeg.exe
  ffmpeg/ffprobe.exe
```

把 `source/` 的内容平铺到 `video_dewatermark/` 后，`runtime/` 与它同级即可。
依赖清单见 `source/README.md`；也可以用自己的 Python 解释器，改 `start.bat` 里的 `PY` 变量。

## 这条流水线做了什么

1. **检测**：抽 48 帧，用「中位帧 Canny ∩ 多帧投票」找固定水印，出现率 > 0.70，输出 bbox + 置信度。
2. **遮罩**：按框取笔画、填洞、膨胀、羽化；判定为半透明叠加层（时间方差衰减 < 0.85）时整框覆盖。
3. **修复**：`cv2.inpaint`（Telea / NS）+ 邻近真实块移植 + 高频纹理回填，rawvideo 管道直编 H.264，保留音轨。
4. **精修**：比对修复区与邻域的纹理方差、边缘密度，只对 `edge_residual` / `over_smooth` 区域换参数重修。
5. **质检**：抽 5 帧，中位数 + 帧通过率聚合，四维加权（纹理 0.35 / 边缘 0.30 / 色彩 0.20 / 自然度 0.15）。
6. **回灌**：不达标就带下一档参数回到遮罩层重跑，最多 2 次，日志里能直接看到「不达标 → 重生成 → 达标」。

## 授权

仅限处理你本人拥有版权或已获得权利人明确授权的视频。禁止用于去除他人作品的水印、台标、版权标识。

## 致谢与出处

示例视频由 `backend/sample_video.py` 程序合成，不含任何第三方素材。