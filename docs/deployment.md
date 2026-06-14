# CallGuard 云端演示部署

CallGuard 的完整能力依赖 TeleAntiFraud、CSEMOTIONS、faster-whisper 和本地 `.joblib` 模型文件。为了避免把大数据和模型权重上传到 GitHub，推荐把部署分为两种模式。

## 1. 本地完整版

适合课程答辩和完整机器学习链路展示。

- ASR 自动转写：可用
- 音频风险识别：可用
- 语音压力识别：可用
- CAEF 融合：可用
- 长音频风险时间线：可用

运行方式：

```powershell
npm run api:dev
npm run web:dev
```

## 2. 云端演示版

适合发给别人在线体验产品外壳。默认不上传大模型和原始数据，重点保留：

- 文本风险识别
- 关键词规则管理
- 历史记录
- Dashboard
- 部署状态页

后端环境变量：

```text
CALLGUARD_DEMO_MODE=1
CALLGUARD_DEPLOYMENT_MODE=cloud-demo
CALLGUARD_DB_PATH=/data/callguard.db
CALLGUARD_CORS_ORIGINS=https://你的前端域名
```

前端环境变量：

```text
NEXT_PUBLIC_API_URL=https://你的后端域名
```

## Docker 本地演示

```powershell
docker compose -f docker-compose.demo.yml up --build
```

打开：

```text
http://127.0.0.1:3000
```

API 健康检查：

```text
http://127.0.0.1:8001/health
```

## Render + Vercel 推荐组合

1. 后端部署到 Render。
2. 使用仓库根目录的 `render.yaml`。
3. 在 Render 中设置 `CALLGUARD_CORS_ORIGINS` 为前端域名。
4. 前端部署到 Vercel。
5. 在 Vercel 中设置 `NEXT_PUBLIC_API_URL` 为 Render 后端地址。

## 为什么云端默认不放完整模型

完整音频链路需要大文件和更高算力，免费云平台容易出现冷启动、磁盘限制和推理超时。课程展示时建议：

- 在线链接展示产品结构、文本风险、规则管理、历史和看板。
- 本地电脑展示 ASR、音频风险、压力识别、CAEF 和风险时间线。

这样既有可访问的产品链接，也保留机器学习大作业的技术深度。
