# CallGuard 通话风险预警系统

CallGuard 是一个面向产品形态的机器学习项目，用于分析通话音频中的诈骗风险、语音压力、情绪状态和可解释风险因素。用户可以上传音频、直接录音，或使用内置 demo 样例，系统会自动转写语音、分析文本风险、分析语音压力，并输出综合预警结果。

项目目标不是“百分百判断对方是不是诈骗”，而是做一个更符合真实场景的早期风险提醒工具：当通话中出现异常压力、诱导转账、冒充身份、验证码、退款理赔等高风险模式时，及时提醒用户冷静核验。

## 快速启动

本项目需要同时启动后端和前端。建议打开两个 PowerShell 窗口，一个跑后端，一个跑前端。

### 1. 启动后端

```powershell
cd C:\Users\22412\Desktop\CallGuard
npm run api:dev
```

看到下面这行就说明后端启动成功：

```text
Uvicorn running on http://127.0.0.1:8001
```

后端 API 地址是：

```text
http://127.0.0.1:8001
```

注意：浏览器直接打开 `http://127.0.0.1:8001/` 出现 `404 Not Found` 是正常的，因为 `8001` 是后端接口，不是网页首页。可以用下面命令检查后端健康状态：

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
```

### 2. 启动前端

再打开一个新的 PowerShell 窗口：

```powershell
cd C:\Users\22412\Desktop\CallGuard
npm run web:dev
```

然后在浏览器打开：

```text
http://127.0.0.1:3000
```

这才是 CallGuard 的网页界面。

### 3. 前端提示端口被占用怎么办

如果出现类似下面的提示：

```text
Port 3000 is in use by process 53784
Another next dev server is already running.
Run taskkill /PID 53784 /F to stop it.
```

说明前端其实已经有一个旧服务在运行。你有两个选择：

直接打开旧服务：

```text
http://127.0.0.1:3000
```

或者关闭旧服务后重新启动：

```powershell
taskkill /PID 53784 /F
npm run web:dev
```

也可以用通用命令关闭占用 `3000` 的进程：

```powershell
Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess | Sort-Object -Unique | ForEach-Object { Stop-Process -Id $_ -Force }
```

如果后端 `8001` 被占用，可以执行：

```powershell
Get-NetTCPConnection -LocalPort 8001 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess | Sort-Object -Unique | ForEach-Object { Stop-Process -Id $_ -Force }
```

## 第一次运行前的环境准备

如果你是在本机继续使用已经配置好的项目，通常不用重复执行本节。如果是换电脑、重新克隆 GitHub 仓库，或删除过依赖，则需要重新安装。

### 1. 安装前端依赖

```powershell
cd C:\Users\22412\Desktop\CallGuard
npm install
```

### 2. 创建 Python 虚拟环境

```powershell
cd C:\Users\22412\Desktop\CallGuard
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r apps\api\requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

### 3. 配置环境变量

复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

不要把真实的 Hugging Face Token 写进 GitHub。需要下载受限数据集时，只在本地 PowerShell 临时设置：

```powershell
$env:HF_TOKEN="你的HuggingFaceToken"
```

## 数据集和模型下载

GitHub 仓库中不会上传原始数据集、大模型、处理后的训练数据和 `.joblib` 模型文件，因为它们体积很大，也可能受数据许可限制。别人克隆项目后，如果想完整复现实验，需要按下面步骤下载和处理。

### 1. TeleAntiFraud 数据集

用途：训练通话诈骗风险识别、音频风险基线、文本风险基线和融合模型。

TeleAntiFraud 是 Hugging Face 上的 gated dataset，需要先申请访问权限：

1. 打开 `https://huggingface.co/datasets/JimmyMa99/TeleAntiFraud`
2. 登录 Hugging Face
3. 接受数据集使用条款
4. 在 Hugging Face 的 `Settings -> Access Tokens` 创建 read token
5. 在 PowerShell 中设置 token：

```powershell
$env:HF_TOKEN="你的HuggingFaceToken"
```

国内网络推荐用镜像下载，项目脚本已经支持 `hf-mirror.com` 和断点续传：

```powershell
cd C:\Users\22412\Desktop\CallGuard
npm run data:download:teleantifraud
```

下载完成后处理数据：

```powershell
npm run data:prepare:teleantifraud
```

处理后会生成：

```text
data/processed/teleantifraud_binary/train.jsonl
data/processed/teleantifraud_binary/test.jsonl
```

如果自动脚本下载失败，也可以手动从网页下载下面文件，放到 `data/raw/teleantifraud/`：

```text
binary_classification.zip
audio.zip
dataset_manifest.json
```

然后再执行：

```powershell
npm run data:prepare:teleantifraud
```

### 2. CSEMOTIONS 数据集

用途：训练中文语音情绪识别和压力感知分支。

国内网络推荐用项目脚本下载：

```powershell
cd C:\Users\22412\Desktop\CallGuard
npm run data:download:csemotions
```

下载后处理 parquet 中的音频：

```powershell
npm run data:prepare:csemotions
```

处理后会生成：

```text
data/interim/csemotions/audio/
data/processed/csemotions_emotion/train.jsonl
data/processed/csemotions_emotion/test.jsonl
```

如果需要手动下载，数据来源是：

```text
https://huggingface.co/datasets/AIDC-AI/CSEMOTIONS
```

手动下载后要保证目录结构类似：

```text
data/raw/csemotions/
  data/
    train-00000-of-00008.parquet
    ...
    train-00007-of-00008.parquet
  README.md
  dataset_infos.json
  NOTICE
```

### 3. ASR 语音转写模型

用途：把通话音频转成中文文本，再交给文本风险模型和规则系统分析。

推荐下载 `faster-whisper-small`：

```powershell
cd C:\Users\22412\Desktop\CallGuard
npm run model:download:asr-small
```

下载位置：

```text
ml/models/asr/faster-whisper-small/
```

项目会优先使用 `small`，如果没有则回退到 `base` 或 `tiny`。

## 训练和复现实验

如果本地已经有训练好的模型文件，可以直接启动产品。如果是从 GitHub 新克隆的仓库，需要重新训练模型。

推荐顺序：

```powershell
npm run ml:train:audio-baseline
npm run ml:train:emotion-baseline
npm run data:asr:teleantifraud
npm run ml:train:text-baseline
npm run ml:train:adaptive-fusion
```

这些命令会生成本地模型文件，例如：

```text
ml/models/audio_baseline/model.joblib
ml/models/emotion_baseline/model.joblib
ml/models/text_baseline/model.joblib
ml/models/fusion/adaptive_fusion.json
```

其中 `.joblib` 文件不会上传到 GitHub，需要本地训练得到。

可以用下面命令检查关键文件是否齐全：

```powershell
npm run check
```

## 项目功能

- 音频上传：支持 `mp3`、`wav`、`m4a` 等常见音频格式。
- 浏览器录音：可以直接在网页中录制一段语音并分析。
- 自动转写：使用本地 faster-whisper 模型进行中文语音识别。
- 文本风险分析：识别诈骗话术、冒充身份、转账、验证码、退款理赔等风险线索。
- 语音压力识别：基于 CSEMOTIONS 训练语音情绪和压力分支。
- 音频风险基线：基于 TeleAntiFraud 训练音频诈骗识别模型。
- 综合融合方法：使用 CallGuard CAEF 方法融合文本、音频、规则和置信度信息。
- 可解释结果：展示风险分数、证据来源、命中因素和安全建议。
- 内置 demo 样例：方便课堂展示，不需要每次手动找音频。

## 当前模型与指标

当前本地实验结果：

```text
音频诈骗识别 baseline:
  accuracy: 0.7175
  macro F1: 0.7142
  fraud recall: 0.8250

文本风险 baseline:
  accuracy: 0.9500
  macro F1: 0.9499

CSEMOTIONS 情绪模型:
  emotion accuracy: 0.5666
  emotion macro F1: 0.5660
  pressure accuracy: 0.6230
  pressure macro F1: 0.6114

CallGuard CAEF 融合方法:
  audio only macro F1: 0.7114
  rules only macro F1: 0.6866
  fixed fusion macro F1: 0.8249
  CAEF macro F1: 0.9246
```

需要注意：在当前数据划分上，文本单模态 baseline 很强，CAEF 的价值主要体现在比固定权重融合更稳，并提供可解释的多源证据融合。报告中应该诚实说明这一点。

## 常用命令

启动后端：

```powershell
npm run api:dev
```

启动前端：

```powershell
npm run web:dev
```

构建前端：

```powershell
npm run web:build
```

检查项目：

```powershell
npm run check
```

下载 TeleAntiFraud：

```powershell
npm run data:download:teleantifraud
```

下载 CSEMOTIONS：

```powershell
npm run data:download:csemotions
```

下载 ASR 模型：

```powershell
npm run model:download:asr-small
```

## 项目结构

```text
apps/
  web/                 Next.js 前端应用
  api/                 FastAPI 后端服务
packages/
  shared/              前后端共享类型
ml/
  configs/             模型和实验配置
  data/                数据集元信息
  notebooks/           探索性实验笔记
  src/callguard_ml/    可复用机器学习代码
scripts/               数据下载、数据处理、模型训练、项目检查脚本
data/
  raw/                 原始下载数据，不上传 GitHub
  interim/             解压或中间处理数据，不上传 GitHub
  processed/           模型可直接使用的数据，不上传 GitHub
  external/            小型外部样例和 demo 配置
docs/                  产品、数据、实验和方法文档
experiments/           实验输出和日志，不上传 GitHub
reports/               课程报告和展示稿草稿
outputs/               最终交付物
infra/                 部署和本地服务配置
```

## GitHub 仓库说明

为了让仓库轻量、安全、合规，下面内容不会上传：

```text
.venv/
node_modules/
data/raw/
data/interim/
data/processed/
ml/models/asr/
ml/models/*/*.joblib
.env
```

所以别人从 GitHub 克隆后，需要自己执行数据下载、数据处理和模型训练命令，才能得到完整本地运行环境。

## 课堂展示建议

推荐展示顺序：

1. 打开 `http://127.0.0.1:3000`，先展示完整产品界面。
2. 点击内置 demo 中的诈骗文本样例，展示文本风险解释。
3. 点击 TeleAntiFraud 音频样例，展示自动转写、音频分析和融合结果。
4. 点击 CSEMOTIONS 高压力样例，展示语音压力分支。
5. 使用浏览器录音功能录一小段中文语音，展示实时产品体验。
6. 最后说明模型不是直接“定罪”，而是做早期预警和风险解释。

## 项目定位

这个项目的核心价值是把机器学习任务做成完整应用，而不是只停留在 notebook 或单个分类器上。它包含数据处理、模型训练、后端接口、前端交互、可解释结果和展示流程，适合作为机器学习大作业中的精品应用方向。
