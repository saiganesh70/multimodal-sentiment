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
## Inference latency (text model, CPU, max_length 48)

| Runtime | Batch 1 median (ms) | Batch 1 p95 (ms) | Batch 32 median (ms) |
|---------|---------------------|------------------|----------------------|
| PyTorch | 39.3 | 44.2 | 698.6 |
| ONNX Runtime | 23.2 | 25.4 | 656.8 |

ONNX output matches PyTorch (max logit difference 2.75e-06, 100% prediction agreement, identical test accuracy and macro-F1). Single run on one laptop CPU.