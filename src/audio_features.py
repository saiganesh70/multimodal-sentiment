import sys, os, subprocess
import numpy as np, pandas as pd
import librosa, imageio_ffmpeg
from joblib import Parallel, delayed
from tqdm import tqdm

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
SR = 16000
N_FEATS = 52  # 20 MFCC mean + 20 MFCC std + 4 pitch + 3 energy + 2 zcr + 2 centroid + 1 duration

def load_audio(path):
    cmd = [FFMPEG, "-v", "quiet", "-i", path, "-vn", "-ac", "1",
           "-ar", str(SR), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32)

def features(path):
    try:
        y = load_audio(path)
        if len(y) < SR // 4:
            return np.zeros(N_FEATS, dtype=np.float32)
        mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20)
        f0 = librosa.yin(y, fmin=50, fmax=400, sr=SR)
        rms = librosa.feature.rms(y=y)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        cent = librosa.feature.spectral_centroid(y=y, sr=SR)[0]
        feats = np.concatenate([
            mfcc.mean(1), mfcc.std(1),
            [f0.mean(), f0.std(), f0.min(), f0.max()],
            [rms.mean(), rms.std(), rms.max()],
            [zcr.mean(), zcr.std()],
            [cent.mean(), cent.std()],
            [len(y) / SR],
        ])
        return feats.astype(np.float32)
    except Exception:
        return np.zeros(N_FEATS, dtype=np.float32)

if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    for split in ["train", "dev", "test"]:
        out = f"data/audio_X_{split}.npy"
        if limit is None and os.path.exists(out):
            print(f"{split}: already done, skipping")
            continue
        paths = pd.read_csv(f"data/{split}_manifest.csv")["video_path"].tolist()
        if limit:
            paths = paths[:limit]
        X = np.array(Parallel(n_jobs=-1)(delayed(features)(p) for p in tqdm(paths, desc=split)))
        bad = int((np.abs(X).sum(1) == 0).sum())
        print(f"{split}: shape={X.shape}, failed/empty clips={bad}")
        if limit is None:
            np.save(out, X)
        else:
            print("sample features:", np.round(X[0][:6], 3))