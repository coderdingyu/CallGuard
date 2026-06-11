# CallGuard Demo Script

## 1. Opening

CallGuard is a call-risk awareness application. It does not claim to detect
lies. Its goal is to warn users when a phone call contains abnormal pressure,
fraud-like wording, or suspicious audio patterns.

## 2. Demo Flow

1. Open the web app at `http://127.0.0.1:3000`.
2. Load a built-in demo sample or record a short voice clip in the browser.
3. Leave the transcript box empty for audio demos to show automatic ASR.
4. Click the analysis button.
5. Explain the result panels:
   - Fusion risk score
   - Audio risk score
   - Text risk score
   - Pressure/emotion auxiliary score
   - ASR transcript
   - Keyword factors
   - Text-model evidence terms
   - Safety suggestion

## 3. Product Message

The key product value is not a single "fraud/not fraud" label. The useful part
is the explainable warning: the user can see why the system is worried and what
to do next.

## 4. Engineering Message

The current version is a complete full-stack prototype:

```text
Next.js frontend
-> FastAPI backend
-> local faster-whisper-small ASR with base/tiny fallback
-> audio-feature Logistic Regression
-> CSEMOTIONS emotion/pressure baseline
-> TF-IDF text Logistic Regression
-> keyword-rule explanation
-> fusion risk score
```

## 5. Limitations

- Audio baseline only analyzes the first 5 seconds.
- Text labels come from audio-level fraud labels, not manually annotated
  transcript labels.
- ASR may make mistakes on noisy calls.
- Browser recording is for local demo use and depends on microphone permission.
- The product should be used as an early-warning assistant, not as final proof.
