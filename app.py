import os, sys
import numpy as np, joblib, gradio as gr, onnxruntime as ort
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from audio_features import features as audio_features
from video_features import features as video_features

LABELS = ["negative", "neutral", "positive"]
MAXLEN = 48
# Weights chosen on the dev set by src/fusion.py (audio got weight 0)
WEIGHTS = {"text": 0.45, "audio": 0.0, "video": 0.55}

tok = AutoTokenizer.from_pretrained("models/text")
text_sess = ort.InferenceSession("models/text.onnx", providers=["CPUExecutionProvider"])
audio_rf = joblib.load("models/audio_rf.joblib")
video_rf = joblib.load("models/video_rf.joblib")

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def as_label(p):
    return None if p is None else {l: float(v) for l, v in zip(LABELS, p)}

def predict(video_path, line):
    if video_path is None:
        raise gr.Error("Please upload a short video clip.")
    probs, notes = {}, []

    if line and line.strip():
        enc = tok([line], truncation=True, max_length=MAXLEN,
                  padding="max_length", return_tensors="np")
        logits = text_sess.run(["logits"], {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64)})[0]
        probs["text"] = softmax(logits)[0]
    else:
        notes.append("No text entered, so the text model was skipped.")

    a = audio_features(video_path)
    if np.abs(a).sum() > 0:
        probs["audio"] = audio_rf.predict_proba(a.reshape(1, -1))[0]
    else:
        notes.append("No usable audio found.")

    v = video_features(video_path)
    if v[16] > 0:
        probs["video"] = video_rf.predict_proba(v.reshape(1, -1))[0]
    else:
        notes.append("No face detected in the sampled frames.")

    if not probs:
        raise gr.Error("None of the three models could read this input.")

    w = {m: WEIGHTS[m] for m in probs}
    if sum(w.values()) == 0:  # e.g. only audio is available
        w = {m: 1.0 for m in probs}
    total = sum(w.values())
    fused = sum(w[m] * probs[m] for m in probs) / total
    notes.append("Fused from: " + ", ".join(f"{m} ({w[m] / total:.0%})" for m in probs if w[m] > 0))

    return (as_label(fused), as_label(probs.get("text")),
            as_label(probs.get("audio")), as_label(probs.get("video")),
            "  \n".join(notes))

with gr.Blocks(title="Multimodal Sentiment (MELD)") as demo:
    gr.Markdown(
        "# Multimodal Sentiment Engine\n"
        "Upload a short clip of a person speaking and type what they say. "
        "Trained on MELD (TV dialogue), so it works best on similar clips. "
        "Fusion = weighted average of text and video (weights tuned on a dev set; "
        "the audio model got weight 0 but is shown for reference).")
    with gr.Row():
        vid = gr.Video(label="Video clip")
        txt = gr.Textbox(label="What is said in the clip", lines=3)
    btn = gr.Button("Predict", variant="primary")
    with gr.Row():
        out_fused = gr.Label(label="Fused", num_top_classes=3)
        out_text = gr.Label(label="Text (DistilBERT, ONNX)", num_top_classes=3)
        out_audio = gr.Label(label="Audio (librosa + RF)", num_top_classes=3)
        out_video = gr.Label(label="Video (HSEmotion + RF)", num_top_classes=3)
    out_note = gr.Markdown()
    btn.click(predict, [vid, txt], [out_fused, out_text, out_audio, out_video, out_note])

if __name__ == "__main__":
    demo.launch()