import numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

LABELS = ["negative", "neutral", "positive"]
L2I = {l: i for i, l in enumerate(LABELS)}

def load(split):
    X = np.load(f"data/audio_X_{split}.npy")
    y = pd.read_csv(f"data/{split}_manifest.csv")["Sentiment"].map(L2I).values
    return X, y

Xtr, ytr = load("train")
Xdv, ydv = load("dev")
Xte, yte = load("test")

model = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                           class_weight="balanced", n_jobs=-1, random_state=42),
)
model.fit(Xtr, ytr)

for name, X, y in [("dev", Xdv, ydv), ("test", Xte, yte)]:
    p = model.predict_proba(X)
    pred = p.argmax(1)
    print(f"{name}: acc={accuracy_score(y, pred):.3f} macro-F1={f1_score(y, pred, average='macro'):.3f}")
    np.save(f"data/audio_probs_{name}.npy", p)

joblib.dump(model, "models/audio_rf.joblib")