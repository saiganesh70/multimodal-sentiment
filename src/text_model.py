import pandas as pd, numpy as np, torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

LABELS = ["negative", "neutral", "positive"]
L2I = {l: i for i, l in enumerate(LABELS)}
MODEL = "distilbert-base-uncased"
device = "cuda" if torch.cuda.is_available() else "cpu"

class TextDS(Dataset):
    def __init__(self, csv, tok):
        df = pd.read_csv(csv)
        self.texts = df["Utterance"].tolist()
        self.y = df["Sentiment"].map(L2I).tolist()
        self.tok = tok
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=48,
                       padding="max_length", return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.y[i])
        return item

def evaluate(model, loader):
    model.eval(); preds, gold = [], []
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            out = model(**{k: v for k, v in b.items() if k != "labels"})
            preds += out.logits.argmax(-1).cpu().tolist()
            gold += b["labels"].cpu().tolist()
    return accuracy_score(gold, preds), f1_score(gold, preds, average="macro")

if __name__ == "__main__":
    print("Using device:", device)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3).to(device)
    train = DataLoader(TextDS("data/train_sent_emo.csv", tok), batch_size=32, shuffle=True)
    dev   = DataLoader(TextDS("data/dev_sent_emo.csv", tok), batch_size=64)
    test  = DataLoader(TextDS("data/test_sent_emo.csv", tok), batch_size=64)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    best = 0
    for epoch in range(2):
        model.train()
        for b in tqdm(train, desc=f"epoch {epoch+1}"):
            b = {k: v.to(device) for k, v in b.items()}
            loss = model(**b).loss
            loss.backward(); opt.step(); opt.zero_grad()
        acc, f1 = evaluate(model, dev)
        print(f"epoch {epoch+1} dev acc={acc:.3f} macro-F1={f1:.3f}")
        if f1 > best:
            best = f1
            model.save_pretrained("models/text"); tok.save_pretrained("models/text")

    model = AutoModelForSequenceClassification.from_pretrained("models/text").to(device)
    print("TEST acc/macro-F1:", evaluate(model, test))