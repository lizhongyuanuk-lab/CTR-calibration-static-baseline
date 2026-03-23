import pandas as pd

df = pd.read_csv("data/Criteo_1M_with_nans.csv")
print("shape:", df.shape)
print("columns:", df.columns.tolist())
print(df.head(3))
print(df.dtypes.head(30))
# 找可能的 label 列
candidates = [c for c in df.columns if c.lower() in ["label", "click", "clicked", "ctr", "y", "target"]]
print("label candidates:", candidates)

# 看每列取值是否像 0/1
for c in candidates:
    print(c, df[c].value_counts().head(5))