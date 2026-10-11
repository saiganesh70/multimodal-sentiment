import numpy as np, pandas as pd, torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ["negative", "neutral", "positive"]
L2I = {l: i for i, l in enumerate(LABELS)}

tok = AutoTokenizer.from_pretrained("models/text")
model = AutoModelForSequenceClassification.from_pretrained("models/text").eval()

for split in ["dev", "test"]:
    df = pd.read_csv(f"data/{split}_manifest.csv")
    texts = df["Utterance"].tolist()
    y = df["Sentiment"].map(L2I).values
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i + 64], truncation=True, max_length=48,
                      padding="max_length", return_tensors="pt")
            probs.append(torch.softmax(model(**enc).logits, -1).numpy())
    p = np.concatenate(probs)
    np.save(f"data/text_probs_{split}.npy", p)
    pred = p.argmax(1)
    print(f"{split}: acc={accuracy_score(y, pred):.3f} macro-F1={f1_score(y, pred, average='macro'):.3f}")