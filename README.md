Website Url:https://f9484a7a9d2ae7bc5e.gradio.live/
---
title: Multimodal Sentiment (MELD)
emoji: 🎭
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
---
# Multimodal Sentiment Engine (MELD)

Sentiment classification (negative / neutral / positive) of TV dialogue using text, audio and video, plus an ablation of late fusion.

## Results (MELD test set, 2,610 utterances)

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| Text only (DistilBERT) | 0.707 | 0.679 |
| Audio only (librosa + Random Forest) | 0.506 | 0.437 |
| Video only (OpenCV + HSEmotion + Random Forest) | 0.438 | 0.421 |
| Fusion: weighted average | 0.708 | 0.678 |
| Fusion: logistic regression | 0.695 | 0.677 |
| Fusion: small MLP | 0.703 | 0.678 |

**Finding:** the transcript carries almost all of the signal. Fusion with dev-tuned weights matched text-only within noise, and the weight search gave audio a weight of 0. Always predicting "neutral" scores about 47% accuracy.

## Pipeline

1. `build_manifest.py` links each CSV row to its video clip.
2. `text_model.py` fine-tunes DistilBERT (2 epochs, CPU).
3. `audio_features.py` / `train_audio.py`: 16 kHz audio via ffmpeg, MFCC / pitch / energy features, Random Forest.
4. `video_features.py` / `train_video.py`: Haar face detection, HSEmotion probabilities on 5 frames per clip, Random Forest.
5. `fusion.py`: weights and combiners are fit on the dev set only; the table above is on test.

## Run it

1. Download MELD (raw videos and CSVs) from https://huggingface.co/datasets/declare-lab/MELD
2. `pip install -r requirements.txt`
3. Run the scripts in `src/` in the order above.

## Limitations

- About 1.5% of clips have no usable audio and a few videos are corrupted; they get zero features.
- Audio and video models use hand-built features, not learned encoders.
- Single run, no confidence intervals; differences under about 1 point are noise.
