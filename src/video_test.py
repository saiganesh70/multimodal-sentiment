import urllib.request
import cv2, numpy as np, pandas as pd
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

fer = HSEmotionRecognizer(model_name="enet_b0_8_best_afew")
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

path = pd.read_csv("data/train_manifest.csv")["video_path"][0]
cap = cv2.VideoCapture(path)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("frames in clip:", n)

for idx in np.linspace(0, max(n - 1, 0), 5).astype(int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    if not ok:
        print(idx, "could not read frame")
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    if len(faces) == 0:
        print(idx, "no face found")
        continue
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)
    emo, scores = fer.predict_emotions(face, logits=False)
    print(idx, emo, np.round(scores, 2))