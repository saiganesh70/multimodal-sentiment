import urllib.request
import sys, os
import numpy as np, pandas as pd, cv2
from joblib import Parallel, delayed
from tqdm import tqdm

N_FRAMES = 5
N_FEATS = 17  # 8 mean probs + 8 max probs + fraction of frames with a face
_fer = None
_cascade = None

def _init():
    global _fer, _cascade
    if _fer is None:
        import urllib.request  # hsemotion-onnx forgets this import
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
        _fer = HSEmotionRecognizer(model_name="enet_b0_8_best_afew")
        _cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def features(path):
    zeros = np.zeros(N_FEATS, dtype=np.float32)
    try:
        _init()
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            return zeros
        probs = []
        for idx in np.linspace(0, n - 1, N_FRAMES).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = _cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            if len(faces) == 0:
                continue
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)
            _, scores = _fer.predict_emotions(face, logits=False)
            probs.append(scores)
        cap.release()
        if not probs:
            return zeros
        P = np.array(probs)
        return np.concatenate([P.mean(0), P.max(0), [len(probs) / N_FRAMES]]).astype(np.float32)
    except Exception:
        return zeros

if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    _init()  # downloads the model file once
    _fer = None  # drop the loaded model so joblib has nothing to pickle
    _cascade = None # makes sure the model file is downloaded before workers start
    n_jobs = max(1, (os.cpu_count() or 4) // 2)
    for split in ["train", "dev", "test"]:
        out = f"data/video_X_{split}.npy"
        if limit is None and os.path.exists(out):
            print(f"{split}: already done, skipping")
            continue
        paths = pd.read_csv(f"data/{split}_manifest.csv")["video_path"].tolist()
        if limit:
            paths = paths[:limit]
        X = np.array(Parallel(n_jobs=n_jobs)(delayed(features)(p) for p in tqdm(paths, desc=split)))
        found = float((X[:, 16] > 0).mean())
        print(f"{split}: shape={X.shape}, clips with at least one face={found:.1%}")
        if limit is None:
            np.save(out, X)