# Results (MELD test set, 2,610 utterances)

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| Text only (DistilBERT) | 0.707 | 0.679 |
| Audio only (librosa + Random Forest) | 0.506 | 0.437 |
| Video only (OpenCV + HSEmotion + Random Forest) | 0.438 | 0.421 |
| Fusion: weighted average | 0.708 | 0.678 |
| Fusion: logistic regression | 0.695 | 0.677 |
| Fusion: small MLP | 0.703 | 0.678 |

Fusion weights and combiners are fit on the dev set only. Fusion matched text-only performance within noise.