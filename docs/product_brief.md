# CallGuard Product Brief

## Positioning

CallGuard is a call-risk awareness application for Chinese phone-call scenarios.
It combines speech emotion, pressure signals, transcription, scam wording, and
explainable scoring to warn users before risky actions such as transfer, code
sharing, or private identity disclosure.

## Target Users

- Students and families who want a visible anti-fraud assistant.
- Older users who may need simple high-risk warnings.
- Course reviewers who need to see a complete machine learning product.

## Core User Flow

1. User uploads a call audio file.
2. The backend creates an analysis task.
3. The ML service performs audio preprocessing, ASR, emotion analysis, text-risk
   classification, and rule extraction.
4. The frontend shows a timeline, transcript, highlights, risk score, and advice.

## MVP Scope

- Upload audio file.
- Show task status.
- Show transcript.
- Show binary fraud-risk classification.
- Show emotion/stress timeline.
- Show explainable warning factors.

## Product Principles

- Do not claim certainty about lying or deception.
- Prefer explainable warnings over black-box verdicts.
- Treat privacy as a product feature.
- Make the interface calm, clear, and trustworthy.

## Demo Narrative

The best demo is a comparison between a normal call and a fraud-like call. The
normal call should show low risk, neutral emotion, and no high-risk phrases. The
fraud-like call should show pressure signals, transfer-related phrases, and a
clear high-risk explanation.

