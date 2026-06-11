# CallGuard 通话风险预警系统

CallGuard 是一个面向真实产品形态的机器学习项目，用于对通话音频进行诈骗风险、语音压力、情绪状态和可解释风险因素分析。用户可以上传音频、录音，或使用内置 demo 样例，系统会自动转写语音、分析文本风险、分析语音压力，并输出一个可解释的综合预警结果。

这个项目的定位不是“百分百判断对方是否诈骗”，而是做一个更安全、更符合实际产品逻辑的早期风险提醒工具：当通话中出现异常压力、诱导转账、冒充身份、验证码、退款理赔等高风险模式时，及时提醒用户冷静核验。

## 如何重新启动项目

每次重新开机、关闭终端、或者想重新演示项目时，按下面步骤启动前后端。

### 1. 打开后端

新开一个 PowerShell 终端，执行：

```powershell
cd C:\Users\22412\Desktop\CallGuard
npm run api:dev
```

后端启动成功后，终端里会看到类似：

```text
Uvicorn running on http://127.0.0.1:8001
```

后端接口地址是：

```text
http://127.0.0.1:8001
```

可以用下面命令检查后端是否正常：

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
```

### 2. 打开前端

再新开一个 PowerShell 终端，执行：

```powershell
cd C:\Users\22412\Desktop\CallGuard
npm run web:dev
```

前端启动成功后，在浏览器打开：

```text
http://127.0.0.1:3000
```

正常情况下，你会看到 CallGuard 的完整网页界面，可以上传音频、浏览内置样例、录音并查看模型分析结果。

### 3. 如果端口被占用

如果提示 `3000` 或 `8001` 端口已经被占用，先关闭旧进程：

```powershell
Get-NetTCPConnection -LocalPort 8001 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess | Sort-Object -Unique | ForEach-Object { Stop-Process -Id $_ -Force }
Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess | Sort-Object -Unique | ForEach-Object { Stop-Process -Id $_ -Force }
```

然后重新执行：

```powershell
npm run api:dev
npm run web:dev
```

注意：这两个命令建议分别放在两个 PowerShell 窗口里运行。

## 第一次运行前的准备

如果你已经在本机成功跑过项目，通常不需要重复这一节。只有在换电脑、重新克隆 GitHub 仓库、删除依赖之后，才需要重新安装环境。

### 前端依赖

```powershell
cd C:\Users\22412\Desktop\CallGuard
npm install
```

### 后端 Python 环境

```powershell
cd C:\Users\22412\Desktop\CallGuard
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r apps\api\requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

### 环境变量

项目默认使用本地后端地址 `http://127.0.0.1:8001`。如果需要复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

不要把真实的 Hugging Face Token 写进 GitHub 仓库。Token 只应该在本地 PowerShell 里临时设置：

```powershell
$env:HF_TOKEN="你的token"
```

## 项目功能

- 音频上传：支持 `mp3`、`wav`、`m4a` 等常见音频格式。
- 浏览器录音：可以直接在网页中录制一段通话片段并分析。
- 自动转写：使用本地 faster-whisper 模型进行中文语音识别。
- 文本风险分析：识别诈骗话术、冒充身份、转账、验证码、退款理赔等风险线索。
- 语音压力识别：基于 CSEMOTIONS 训练语音情绪和压力分支。
- 音频风险基线：基于 TeleAntiFraud 训练音频诈骗识别模型。
- 综合融合方法：使用 CallGuard CAEF 方法融合文本、音频、规则和置信度信息。
- 可解释结果：展示风险分数、证据来源、命中因素和安全建议。
- 内置 demo 样例：方便课堂展示，不需要每次手动找音频。

## 当前模型与数据

本项目目前使用的数据和模型包括：

- TeleAntiFraud：用于通话诈骗风险识别。
- CSEMOTIONS：用于中文语音情绪和压力状态识别。
- faster-whisper-small：用于本地中文语音转写。
- 文本 TF-IDF + Logistic Regression：作为文本风险基线。
- 音频特征 + Logistic Regression：作为音频风险基线。
- CallGuard CAEF：作为本项目提出的可解释融合方法。

模型文件和原始数据体积较大，默认不会上传到 GitHub。它们保存在本地：

```text
data/raw/
data/interim/
data/processed/
ml/models/
```

其中 `.joblib`、ASR 模型、大型数据集都已经通过 `.gitignore` 排除，避免 GitHub 仓库过大或泄露数据。

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

检查项目关键文件和模型状态：

```powershell
npm run check
```

下载更大的 ASR 模型：

```powershell
npm run model:download:asr-small
```

处理 TeleAntiFraud 数据：

```powershell
npm run data:prepare:teleantifraud
```

处理 CSEMOTIONS 数据：

```powershell
npm run data:prepare:csemotions
```

训练音频基线：

```powershell
npm run ml:train:audio-baseline
```

训练情绪/压力基线：

```powershell
npm run ml:train:emotion-baseline
```

训练文本基线：

```powershell
npm run ml:train:text-baseline
```

训练融合方法：

```powershell
npm run ml:train:adaptive-fusion
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
scripts/               数据处理、模型训练、项目检查脚本
data/
  raw/                 原始下载数据
  interim/             解压或中间处理数据
  processed/           模型可直接使用的数据
  external/            外部参考或演示样例
docs/                  产品、数据、实验和方法文档
experiments/           实验输出和日志
reports/               课程报告和展示稿草稿
outputs/               最终交付物
infra/                 部署和本地服务配置
```

## 课堂展示建议

推荐展示顺序：

1. 打开网页首页，先展示完整产品界面。
2. 点击内置 demo 中的诈骗文本样例，展示文本风险解释。
3. 点击 TeleAntiFraud 音频样例，展示自动转写、音频分析和融合结果。
4. 点击 CSEMOTIONS 高压力样例，展示语音压力分支。
5. 使用浏览器录音功能录一小段中文语音，展示实时产品体验。
6. 最后说明模型不是直接“定罪”，而是做早期预警和风险解释。

## 项目定位

这个项目的核心价值是把机器学习任务做成一个完整产品，而不是只停留在 notebook 或单个分类器上。它包含数据处理、模型训练、后端接口、前端交互、可解释结果和演示流程，适合作为机器学习大作业中的精品应用方向。
