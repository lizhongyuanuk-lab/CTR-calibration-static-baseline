# =========================
# train.py —— CTR 预测：LR baseline + 预处理流水线
# =========================

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# =========================
# 配置区：路径 & 列名
# =========================

DATA_PATH = "data/Criteo_1M_with_nans.csv"   # 原始数据
LABEL_COL = "target"                         # 标签列名

MODEL_PATH = "models/lr_pipeline.joblib"
METRICS_PATH = "outputs/metrics.txt"

# 新增：切分文件保存路径
TRAIN_PATH = "data/train.csv"
VALID_PATH = "data/valid.csv"
TEST_PATH = "data/test.csv"


def main():
    # =========================
    # 0) I/O dirs：确保输出文件夹存在
    # =========================
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # =========================
    # 1) Load：读取数据
    # =========================
    df = pd.read_csv(DATA_PATH)

    if LABEL_COL not in df.columns:
        raise ValueError(
            f"Label column '{LABEL_COL}' not found. Columns: {df.columns.tolist()[:30]} ..."
        )

    # =========================
    # 2) Split X/y：拆分特征 X 和标签 y
    # =========================
    X = df.drop(columns=[LABEL_COL])
    y = df[LABEL_COL].astype(int)

    click_rate = float(y.mean())
    print(f"[INFO] rows={len(df):,}, cols={df.shape[1]}, click_rate={click_rate:.4f}")

    # =========================
    # 3) Train/valid/test split
    # =========================
    # 先切 test（20%）
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 再从剩下的 80% 里切出 valid（总量 20%）
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp,
        test_size=0.25,   # 0.25 * 0.8 = 0.2
        random_state=42,
        stratify=y_temp
    )

    print(f"[INFO] train={len(X_train):,}, valid={len(X_valid):,}, test={len(X_test):,}")

    # =========================
    # 4) 保存切分结果
    # =========================
    train_df = X_train.copy()
    train_df[LABEL_COL] = y_train.values
    train_df.to_csv(TRAIN_PATH, index=False)

    valid_df = X_valid.copy()
    valid_df[LABEL_COL] = y_valid.values
    valid_df.to_csv(VALID_PATH, index=False)

    test_df = X_test.copy()
    test_df[LABEL_COL] = y_test.values
    test_df.to_csv(TEST_PATH, index=False)

    print(f"[INFO] saved split files -> {TRAIN_PATH}, {VALID_PATH}, {TEST_PATH}")

    # =========================
    # 5) Column types：识别数值列/类别列
    # =========================
    num_cols = [c for c in X.columns if c.startswith("intCol_")]
    cat_cols = [c for c in X.columns if c.startswith("catCol_")]

    if len(num_cols) == 0 and len(cat_cols) == 0:
        num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    print(f"[INFO] num_cols={len(num_cols)}, cat_cols={len(cat_cols)}")

    # =========================
    # 6) Preprocess pipelines
    # =========================
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    # =========================
    # 7) Model
    # =========================
    clf = LogisticRegression(
        solver="liblinear",
        max_iter=200,
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])

    # =========================
    # 8) Fit：训练
    # =========================
    model.fit(X_train, y_train)

    # =========================
    # 9) Evaluate：先在 test 上看 baseline
    # =========================
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    ll = log_loss(y_test, y_prob)

    print(f"[RESULT] AUC={auc:.6f}  LogLoss={ll:.6f}")

    # =========================
    # 10) Save：保存模型 & 指标
    # =========================
    joblib.dump(model, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(f"rows={len(df)}\n")
        f.write(f"click_rate={click_rate:.6f}\n")
        f.write(f"train_rows={len(X_train)}\n")
        f.write(f"valid_rows={len(X_valid)}\n")
        f.write(f"test_rows={len(X_test)}\n")
        f.write(f"AUC={auc:.6f}\n")
        f.write(f"LogLoss={ll:.6f}\n")

    print(f"[INFO] saved model -> {MODEL_PATH}")
    print(f"[INFO] saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()