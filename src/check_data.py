import pandas as pd

df = pd.read_csv("data/train_sent_emo.csv")
print(df[["Utterance", "Sentiment"]].head())
print(df["Sentiment"].value_counts())
print("rows:", len(df))