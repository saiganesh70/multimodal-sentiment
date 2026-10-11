import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

LABELS = ["negative", "neutral", "positive"]
L2I = {l: i for i, l in enumerate(LABELS)}
MODS = ["text", "audio", "video"]

y = {s: pd.read_csv(f"data/{s}_manifest.csv")["Sentiment"].map(L2I).values
     for s in ["dev", "test"]}
P = {s: {m: np.load(f"data/{m}_probs_{s}.npy") for m in MODS} for s in ["dev", "test"]}

def score(yy, p):
    pred = p.argmax(1)
    return accuracy_score(yy, pred), f1_score(yy, pred, average="macro")

rows = []
for m in MODS:
    rows.append((f"{m} only", *score(y["test"], P["test"][m])))

# 1) Weighted late fusion: weights chosen on DEV, reported on TEST
best_f1, best_w = -1, None
grid = np.round(np.arange(0, 1.01, 0.05), 2)
for wt in grid:
    for wa in grid:
        wv = round(1 - wt - wa, 2)
        if wv < 0:
            continue
        p = wt * P["dev"]["text"] + wa * P["dev"]["audio"] + wv * P["dev"]["video"]
        f1 = score(y["dev"], p)[1]
        if f1 > best_f1:
            best_f1, best_w = f1, (wt, wa, wv)
wt, wa, wv = best_w
print(f"weights chosen on dev: text={wt} audio={wa} video={wv}")
p = wt * P["test"]["text"] + wa * P["test"]["audio"] + wv * P["test"]["video"]
rows.append(("Fusion: weighted average", *score(y["test"], p)))

# 2) Learned fusion on concatenated probabilities (fit on DEV, report on TEST)
Xdv = np.hstack([P["dev"][m] for m in MODS])
Xte = np.hstack([P["test"][m] for m in MODS])
lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(Xdv, y["dev"])
rows.append(("Fusion: logistic regression", *score(y["test"], lr.predict_proba(Xte))))
mlp = MLPClassifier(hidden_layer_sizes=(16,), alpha=1e-2, max_iter=2000,
                    random_state=42).fit(Xdv, y["dev"])
rows.append(("Fusion: small MLP", *score(y["test"], mlp.predict_proba(Xte))))

print("\n| Model | Accuracy | Macro-F1 |")
print("|-------|----------|----------|")
for name, acc, f1 in rows:
    print(f"| {name} | {acc:.3f} | {f1:.3f} |")