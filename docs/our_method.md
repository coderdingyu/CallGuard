# CallGuard Proposed Method: CAEF

## Method Name

CAEF stands for Confidence-Aware Explainable Fusion.

The motivation is that a fixed multimodal weight is unsafe when component
models have very different reliability. In the current data, the text model is
much stronger than the handcrafted audio baseline. A fixed 65% audio weight
therefore reduces performance.

## Inputs

CAEF receives three component signals:

- audio fraud probability
- text-model fraud probability
- explainable keyword-rule score

Text classification preserves the original ASR surface text. The rule branch
uses Unicode normalization and Traditional-to-Simplified Chinese conversion.
This dual-path processing keeps useful model features while improving rule
coverage for Traditional Chinese ASR output.

## Adaptive Weighting

For a probability `p`, confidence is:

```text
confidence(p) = 2 * abs(p - 0.5)
```

Each modality receives an unnormalized weight:

```text
weight_m = reliability_m ^ gamma
           * (confidence_floor + (1 - confidence_floor) * confidence_m)
```

The weights are normalized before fusion. When text confidence exceeds the
learned gate, CAEF trusts the stronger text model and suppresses the weaker
audio branch.

The rule branch can provide an upward risk adjustment. Cross-validation selected
a zero rule adjustment for the current small dataset, so rules are currently
used for explanation rather than changing the final score.

## Experimental Protocol

The method uses a disjoint three-part protocol:

```text
160 ASR samples: train the validation text component
80 ASR samples: select CAEF parameters
80 ASR samples: independent final test
```

For validation, the audio model is trained on 3920 TeleAntiFraud training
samples after excluding the 80 fusion-validation audio files.

## Results

| Method | Accuracy | Macro F1 | Fraud Recall |
|---|---:|---:|---:|
| Audio only | 0.7125 | 0.7114 | 0.775 |
| Rules only | 0.7125 | 0.6866 | 0.425 |
| Text only | 0.9500 | 0.9499 | 0.900 |
| Fixed 65/35 fusion | 0.8250 | 0.8249 | 0.800 |
| CallGuard CAEF | 0.9250 | 0.9246 | 0.850 |

CAEF improves Macro F1 by about 0.10 over fixed fusion. It does not exceed the
strongest text-only model because the current audio baseline remains weaker.
This is an important experimental finding rather than something to hide:
multimodal fusion is only useful when modality reliability is handled carefully.

## Current Learned Parameters

```text
audio reliability: 0.6732
text reliability: 1.0000
reliability power: 3.0
text confidence gate: 0.15
confidence floor: 0.60
decision threshold: 0.525
```

The product API loads these parameters from:

```text
ml/models/fusion/adaptive_fusion.json
```

## Product Case Study

One kidnapping-style fraud sample produced an uncertain text prediction but a
high-confidence audio prediction:

```text
text-model score: 0.4907
audio-model score: 0.9628
text confidence: 0.0186
audio confidence: 0.9256
adaptive audio weight: 0.3276
adaptive text weight: 0.6724
final CAEF score: 0.6454
final prediction: fraud
```

The text-only threshold would miss this sample. CAEF enables the audio branch
only because the text model is uncertain, correcting the result without applying
a large audio weight to every call.
