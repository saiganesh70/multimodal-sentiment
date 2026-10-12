import os
import pandas as pd

ROOT = r"C:\Users\User\MELD_raw\MELD.Raw"
SPLITS = {
    "train": ("train_sent_emo.csv", "train_splits"),
    "dev":   ("dev_sent_emo.csv",   "dev_splits_complete"),
    "test":  ("test_sent_emo.csv",  "output_repeated_splits_test"),
}

for name, (csv, folder) in SPLITS.items():
    df = pd.read_csv(f"data/{csv}")
    df["video_path"] = [
        os.path.join(ROOT, folder, f"dia{d}_utt{u}.mp4")
        for d, u in zip(df["Dialogue_ID"], df["Utterance_ID"])
    ]
    df["exists"] = df["video_path"].map(os.path.exists)
    print(f"{name}: {len(df)} rows, {int(df['exists'].sum())} with a video file")
    keep = df[df["exists"]][["Utterance", "Sentiment", "Dialogue_ID", "Utterance_ID", "video_path"]]
    keep.to_csv(f"data/{name}_manifest.csv", index=False)