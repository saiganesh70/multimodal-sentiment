import time
import numpy as np, pandas as pd, torch, onnxruntime as ort
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MAXLEN = 48
LABELS = ["negative", "neutral", "positive"]
L2I = {l: i for i, l in enumerate(LABELS)}

tok = AutoTokenizer.from_pretrained("models/text")
model = AutoModelForSequenceClassification.from_pretrained(
    "models/text", attn_implementation="eager").eval()

class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

wrapped = Wrap(model).eval()

# ---- 1) Export ----
dummy = tok(["this is a test", "another one"], truncation=True, max_length=MAXLEN,
            padding="max_length", return_tensors="pt")
torch.onnx.export(
    wrapped, (dummy["input_ids"], dummy["attention_mask"]), "models/text.onnx",
    input_names=["input_ids", "attention_mask"], output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"},
                  "logits": {0: "batch"}},
    opset_version=17, dynamo=False)
print("exported models/text.onnx")

sess = ort.InferenceSession("models/text.onnx", providers=["CPUExecutionProvider"])

def run_onnx(enc):
    return sess.run(["logits"], {"input_ids": enc["input_ids"].numpy(),
                                 "attention_mask": enc["attention_mask"].numpy()})[0]

def run_torch(enc):
    with torch.no_grad():
        return wrapped(enc["input_ids"], enc["attention_mask"]).numpy()

# ---- 2) Correctness on the full test set ----
df = pd.read_csv("data/test_manifest.csv")
texts = df["Utterance"].tolist()
y = df["Sentiment"].map(L2I).values
lo_t, lo_o = [], []
for i in range(0, len(texts), 64):
    enc = tok(texts[i:i + 64], truncation=True, max_length=MAXLEN,
              padding="max_length", return_tensors="pt")
    lo_t.append(run_torch(enc))
    lo_o.append(run_onnx(enc))
lo_t, lo_o = np.concatenate(lo_t), np.concatenate(lo_o)
print(f"max abs logit difference: {np.abs(lo_t - lo_o).max():.2e}")
print(f"prediction agreement: {(lo_t.argmax(1) == lo_o.argmax(1)).mean():.4f}")
for name, lo in [("PyTorch", lo_t), ("ONNX", lo_o)]:
    pred = lo.argmax(1)
    print(f"{name}: acc={accuracy_score(y, pred):.3f} macro-F1={f1_score(y, pred, average='macro'):.3f}")

# ---- 3) Latency benchmark (CPU) ----
def bench(fn, enc, n):
    for _ in range(10):
        fn(enc)
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn(enc)
        times.append((time.perf_counter() - t) * 1000)
    return np.median(times), np.percentile(times, 95)

enc1 = tok(texts[:1], truncation=True, max_length=MAXLEN, padding="max_length", return_tensors="pt")
enc32 = tok(texts[:32], truncation=True, max_length=MAXLEN, padding="max_length", return_tensors="pt")
print("\n| Runtime | batch 1 median (ms) | batch 1 p95 (ms) | batch 32 median (ms) |")
print("|---------|---------------------|------------------|----------------------|")
for name, fn in [("PyTorch (CPU)", run_torch), ("ONNX Runtime (CPU)", run_onnx)]:
    m1, p1 = bench(fn, enc1, 200)
    m32, _ = bench(fn, enc32, 30)
    print(f"| {name} | {m1:.1f} | {p1:.1f} | {m32:.1f} |")