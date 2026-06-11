# Data Plan

## Priority Datasets

| Dataset | Modality | Language | Purpose | Status |
| --- | --- | --- | --- | --- |
| TeleAntiFraud | audio + text | Chinese | Main fraud/normal training data | prepared |
| CSEMOTIONS | audio | Chinese | Speech emotion and pressure auxiliary modeling | prepared |
| ESD | audio | Mandarin + English | Optional speech emotion pretraining | planned |
| RAVDESS | audio | English | Quick emotion baseline | planned |
| CREMA-D | audio/video | English | Emotion robustness | planned |
| ChiFraud | text | Chinese | Text fraud-risk augmentation | planned |

## Local Data Layout

```text
data/raw/<dataset_name>/       Original downloaded files.
data/interim/<dataset_name>/   Extracted and normalized intermediate files.
data/processed/<task_name>/    Final train/validation/test files.
```

## Unified Labels

### Emotion Labels

```text
neutral
happy
sad
angry
fearful
surprise
playfulness
```

### Risk Labels

```text
normal
low_risk
medium_risk
high_risk
fraud
```

For the first baseline, TeleAntiFraud binary labels will be mapped as:

```text
normal -> normal
fraud  -> fraud
```

## Initial Processing Goals

1. Download or verify raw dataset files.
2. Extract archives into `data/interim`.
3. Build a clean metadata file with audio path, text/instruction, label, split,
   dataset name, and task name.
4. Export model-ready JSONL and CSV files.
5. Keep licenses and dataset source notes in `ml/data/dataset_registry.csv`.

## Current TeleAntiFraud Status

The binary fraud detection subset has been prepared locally:

```text
train records: 4000
test records: 400
train labels: normal 2000, fraud 2000
test labels: normal 200, fraud 200
referenced audio paths: 4400 unique paths
missing referenced audio files: 0
```

The full audio archive contains more files than the binary subset uses. This is
expected because the archive also supports other TeleAntiFraud tasks.

## Current CSEMOTIONS Status

CSEMOTIONS has been prepared locally:

```text
records: 4160
train records: 3327
test records: 833
labels: neutral, happy, angry, sad, surprise, playfulness, fearful
feature window: first 5 seconds
exported model: ml/models/emotion_baseline/model.joblib
```

The pressure branch maps emotion labels into an auxiliary state signal:

```text
normal pressure: neutral, happy, playfulness
medium pressure: sad, surprise
high pressure: fearful, angry
```

This branch is used for product awareness and explanation, not as a direct fraud
label.
