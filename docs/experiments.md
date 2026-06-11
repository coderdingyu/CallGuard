# Experiments

## Audio Baseline: Logistic Regression on Handcrafted Features

This baseline uses traditional speech/audio features from the first 5 seconds of
each call clip:

- duration
- RMS energy
- zero-crossing rate
- spectral centroid
- spectral bandwidth
- spectral rolloff
- MFCC mean and standard deviation

The classifier is a balanced Logistic Regression model with feature
standardization.

```text
dataset: TeleAntiFraud binary fraud detection
train samples: 4000
test samples: 400
accuracy: 0.7175
macro F1: 0.7142
normal precision: 0.7771
normal recall: 0.6100
normal F1: 0.6835
fraud precision: 0.6790
fraud recall: 0.8250
fraud F1: 0.7449
feature extraction failures: 0
```

This is the first traditional machine learning baseline. It is intentionally
lightweight and interpretable. The next model should use pretrained speech
representations such as wav2vec2 or HuBERT to improve performance and reduce
manual feature engineering.

The model has higher fraud recall than normal recall, which means it tends to be
cautious and may over-warn on normal calls. This is acceptable for an early risk
warning baseline, but later product versions should tune thresholds and fusion
logic to reduce false positives.

The product API loads the exported model from:

```text
ml/models/audio_baseline/model.joblib
```

## Emotion and Pressure Baseline: CSEMOTIONS

CSEMOTIONS is used to add an auxiliary speech-state branch. The model uses the
same handcrafted audio feature extractor as the fraud audio baseline and trains
two Logistic Regression classifiers:

- a seven-class emotion classifier
- a three-class pressure classifier derived from emotion groups

The pressure branch is deliberately treated as an auxiliary state cue. It is
shown in the product UI but is not fused into the final fraud decision.

```text
dataset: CSEMOTIONS Mandarin emotional speech
train samples: 3327
test samples: 833
emotion labels: neutral, happy, angry, sad, surprise, playfulness, fearful
emotion accuracy: 0.5666
emotion macro F1: 0.5660
pressure labels: normal, medium, high
pressure accuracy: 0.6230
pressure macro F1: 0.6114
feature extraction failures: 0
export path: ml/models/emotion_baseline/model.joblib
```

The pressure score is computed from the dedicated pressure model probabilities:

```text
normal -> 0.18
medium -> 0.62
high -> 0.925
```

The frontend now displays pressure score, primary emotion, emotion confidence,
pressure probabilities, and the top emotion contributors.

## Fusion Baseline: Audio + Text Risk

The current product endpoint combines the exported audio baseline with the
text-risk score:

```text
audio only: audio score
text only: text risk score
audio + text: 0.65 * audio score + 0.35 * text score
```

In a live API test, a TeleAntiFraud fraud audio sample plus a high-risk transcript
produced:

```text
fusion risk score: 0.6025
risk level: medium
fusion weights: audio 0.65, text 0.35
```

This gives the application a full multimodal baseline before adding a pretrained
speech-risk model.

## Text Baseline: TF-IDF on ASR Transcripts

TeleAntiFraud binary metadata does not provide clean manual transcripts in the
processed split, so the current text dataset is built by running local ASR over a
balanced subset of the audio data.

```text
dataset: TeleAntiFraud ASR transcripts
ASR model: faster-whisper-base
train samples: 240
test samples: 80
model: TF-IDF character n-gram + Logistic Regression
accuracy: 0.9500
macro F1: 0.9499
weighted F1: 0.9499
export path: ml/models/text_baseline/model.joblib
```

This model is intentionally compact and classical, which makes it easy to train
locally and explain in a course report. The current product text score is:

```text
max(0.8 * text model fraud probability + 0.2 * keyword-rule score,
    0.85 * keyword-rule score)
```

The limitation is that the text labels inherit audio-level fraud labels and ASR
errors. A stronger next version should use more ASR samples, manually corrected
transcripts, or a pretrained Chinese text model.

## ASR MVP: faster-whisper-small

The product now includes local ASR:

```text
ASR model: faster-whisper-small
fallback models: faster-whisper-base, faster-whisper-tiny
device: CPU
compute type: int8
language hint: zh
model path: ml/models/asr/faster-whisper-small
```

If users upload audio without a manual transcript, `/api/analyze/call`
automatically generates a transcript and feeds it into the existing text-risk
rules. Manual transcript input still works as an override or correction path.
The displayed transcript is converted to Simplified Chinese, while the original
ASR surface text can still be preserved internally for model compatibility.

This completes the current product loop:

```text
audio upload / browser recording / built-in demo sample
-> ASR transcript
-> audio baseline score
-> emotion/pressure auxiliary score
-> text rule score
-> fused risk score
-> frontend explanation
```

In a local API test with a TeleAntiFraud fraud audio sample, the small ASR model
produced a usable Mandarin transcript containing bank/customer-service wording.
The fused endpoint returned:

```text
ASR model: faster-whisper-small
transcript source: asr
fusion risk score: 0.5303
risk level: medium
text factors: 客服, 银行, 现在
```

The larger ASR model is downloaded locally with:

```powershell
npm run model:download:asr-small
```

## Proposed Method: CallGuard CAEF

The project method is Confidence-Aware Explainable Fusion. Unlike fixed 65/35
fusion, it estimates component reliability on a disjoint validation split and
changes audio/text weights according to per-sample confidence.

The final protocol uses:

```text
fusion text fit: 160 samples
fusion validation: 80 samples
independent test: 80 samples
audio validation fit: 3920 samples
```

Final comparison:

```text
audio only macro F1: 0.7114
rules only macro F1: 0.6866
text only macro F1: 0.9499
fixed 65/35 macro F1: 0.8249
CallGuard CAEF macro F1: 0.9246
```

CAEF is substantially better than fixed fusion but remains slightly below the
strongest text-only baseline. The result shows why reliability-aware gating is
necessary when one modality is weaker. Full equations and learned parameters
are documented in `docs/our_method.md`.

In an online product test, CAEF corrected a text-model false negative:

```text
text score: 0.4907
audio score: 0.9628
adaptive weights: audio 0.3276, text 0.6724
CAEF score: 0.6454
prediction: fraud
```
