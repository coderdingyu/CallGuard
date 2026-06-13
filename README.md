# CallGuard 通话风险预警系统

CallGuard 是一个面向中文通话场景的机器学习全栈应用。它可以上传音频、浏览器录音或输入通话文本，自动完成语音转写、诈骗话术识别、语音压力/情绪分析和可解释融合预警。

项目定位不是“百分百判断诈骗”，而是做早期风险提醒：当通话中出现转账、验证码、冒充身份、紧急施压、退款理赔等模式时，提醒用户暂停操作并通过官方渠道核验。

## 本地快速启动

进入项目根目录：

```powershell
cd CallGuard
```

启动后端：

```powershell
npm run api:dev
```

启动前端：

```powershell
npm run web:dev
```

浏览器打开：

```text
http://127.0.0.1:3000
```

后端地址是：

```text
http://127.0.0.1:8001
```

直接打开 `8001/` 看到 `404 Not Found` 是正常的，因为它是 API 服务。可以用下面命令检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
```

如果前端提示 `Another next dev server is already running`，说明 `3000` 已经有一个前端服务在运行，直接打开 `http://127.0.0.1:3000` 即可。也可以按提示执行：

```powershell
taskkill /PID 进程号 /F
```

## 平台化 MVP 功能

本项目已经从单页 demo 升级为本地精品应用，包含四个主要视图：

- 分析台：上传音频、录音、选择 demo、输入文本并查看融合风险结果。
- 风险时间线：长音频会按 5 秒窗口分段分析，定位通话中风险升高的时间位置。
- 数据看板：查看总分析次数、高风险次数、平均风险分、风险分布和高频风险关键词。
- 历史记录：自动保存每次分析摘要，可查看详情或删除。
- 规则管理：管理文本风险关键词，默认规则可停用，自定义规则可新增和删除。

历史记录和规则保存在本地 SQLite：

```text
data/local/callguard.db
```

该文件不会上传 GitHub。

## 第一次运行前准备

安装前端依赖：

```powershell
npm install
```

创建 Python 虚拟环境并安装后端依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r apps\api\requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

不要把真实 Hugging Face Token 写入 GitHub。需要下载受限数据集时，只在当前 PowerShell 中临时设置：

```powershell
$env:HF_TOKEN="你的HuggingFaceToken"
```

## 数据集与模型

GitHub 仓库不会上传原始数据、大模型、处理后的训练数据和 `.joblib` 模型文件。完整复现实验需要在本地下载和训练。

下载 TeleAntiFraud：

```powershell
npm run data:download:teleantifraud
npm run data:prepare:teleantifraud
```

下载 CSEMOTIONS：

```powershell
npm run data:download:csemotions
npm run data:prepare:csemotions
```

下载本地 ASR 模型：

```powershell
npm run model:download:asr-small
```

推荐训练顺序：

```powershell
npm run ml:train:audio-baseline
npm run ml:train:emotion-baseline
npm run data:asr:teleantifraud
npm run ml:train:text-baseline
npm run ml:train:adaptive-fusion
```

## 常用命令

```powershell
npm run api:dev
npm run web:dev
npm run web:build
npm run check
npm run test:api
```

## 当前模型指标

```text
音频诈骗识别 baseline:
  accuracy: 0.7175
  macro F1: 0.7142
  fraud recall: 0.8250

文本风险 baseline:
  accuracy: 0.9500
  macro F1: 0.9499

CSEMOTIONS 情绪/压力模型:
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

当前数据划分上，文本单模态 baseline 很强；CAEF 的价值主要体现在比固定权重融合更稳，并提供多源证据解释。

## 与 CallGuardAI 的差异

CallGuardAI 的产品外壳更接近 hackathon 平台，强调部署、历史、看板、认证和多语言包装。CallGuard 的核心优势是中文场景、真实数据集、可复现训练链路和自定义融合方法。

本项目现在补齐了 CallGuardAI 式产品外壳中的关键部分：历史记录、数据看板和规则管理；同时保留 TeleAntiFraud、CSEMOTIONS、faster-whisper、文本 baseline、音频 baseline 和 CAEF 融合方法。

## 项目结构

```text
apps/
  web/                 Next.js 前端应用
  api/                 FastAPI 后端服务
packages/
  shared/              前后端共享类型
ml/
  configs/             模型和实验配置
  src/callguard_ml/    可复用机器学习代码
scripts/               数据下载、处理、训练和测试脚本
data/
  raw/                 原始下载数据，不上传 GitHub
  interim/             中间处理数据，不上传 GitHub
  processed/           训练数据，不上传 GitHub
  local/               SQLite 本地数据库，不上传 GitHub
docs/                  产品、数据、实验和方法文档
reports/               课程报告和展示稿草稿
```

## 课堂展示建议

1. 打开分析台，选择内置诈骗文本 demo，展示文本风险解释。
2. 选择 TeleAntiFraud 音频 demo，展示 ASR、音频风险和 CAEF 融合。
3. 选择 CSEMOTIONS 高压力 demo，展示语音压力分支。
4. 切换到历史记录，展示分析结果已自动保存。
5. 切换到数据看板，展示风险分布和 Top 风险关键词。
6. 切换到规则管理，新增一个关键词并说明规则可运营。
7. 上传或选择一段较长音频，展示 5 秒分段风险时间线，说明系统不仅给整体分数，还能定位风险升高的片段。

这个项目的价值在于把机器学习任务做成完整应用，而不是只停留在 notebook 或单个分类器上。
