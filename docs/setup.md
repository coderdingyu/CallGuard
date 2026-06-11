# Setup

## Python Environment

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip huggingface_hub datasets pandas pyarrow -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

## TeleAntiFraud Access

TeleAntiFraud is a gated Hugging Face dataset. Before command-line download:

1. Open <https://huggingface.co/datasets/JimmyMa99/TeleAntiFraud>.
2. Log in.
3. Accept the dataset access conditions.
4. Create a read token from `Settings -> Access Tokens`.
5. Set the token locally:

```powershell
$env:HF_TOKEN="hf_xxx"
```

Then download the priority files:

```powershell
.\.venv\Scripts\python.exe scripts\download_datasets.py --dataset teleantifraud --method curl
```

If the Hugging Face CLI works on the current network, this also works:

```powershell
.\.venv\Scripts\python.exe scripts\download_datasets.py --dataset teleantifraud
```

After downloading:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_teleantifraud.py
```

## CSEMOTIONS Emotion Data

CSEMOTIONS is used for the auxiliary speech emotion and pressure branch. After
downloading it from ModelScope into `data/raw/csemotions`, prepare and train it
with:

```powershell
npm run data:prepare:csemotions
npm run ml:train:emotion-baseline
```

The prepare step expands embedded WAV bytes from parquet into:

```text
data/interim/csemotions/audio
data/processed/csemotions_emotion
```

The exported product model is:

```text
ml/models/emotion_baseline/model.joblib
```

## Larger Local ASR

The product prefers `faster-whisper-small` for higher-quality Mandarin
transcription and falls back to base or tiny when the larger model is absent.

```powershell
npm run model:download:asr-small
```

The downloaded model is stored at:

```text
ml/models/asr/faster-whisper-small
```

## Local Product Demo

Run the backend and frontend in two terminals:

```powershell
npm run api:dev
npm run web:dev
```

The frontend opens at:

```text
http://127.0.0.1:3000
```

The backend API uses:

```text
http://127.0.0.1:8001
```

The frontend includes browser microphone recording and built-in text/audio demo
samples. Microphone recordings are encoded as WAV in the browser so the local
audio pipeline does not require a system FFmpeg installation.

## Manual Download Fallback

If command-line download is blocked, download these files from the web page and
place them in `data/raw/teleantifraud/`:

```text
binary_classification.zip
audio.zip
dataset_manifest.json
```

Then run the same prepare command.

## Rebuild Text Baseline

The text baseline is trained from local ASR transcripts:

```powershell
npm run data:asr:teleantifraud
npm run ml:train:text-baseline
npm run ml:train:adaptive-fusion
```

The ASR step is resumable and can take several minutes because it transcribes
audio locally. The adaptive-fusion step caches audio features, uses a disjoint
validation split, and exports the CAEF configuration used by the product API.
