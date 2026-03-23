# calibrate.py
# 作用：
# 1. 读取原始训练好的模型
# 2. 读取 validation 数据集
# 3. 在 validation 上拟合两种校准器：
#    - Platt scaling (sigmoid)
#    - Isotonic regression
# 4. 保存校准后的模型
#
# 注意：
# 这里“拟合校准器”用的是 valid.csv
# 最终效果评估应该去 test.csv 上看，而不是在 valid 上作为最终结果

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV 
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

from config import (
    VALID_PATH,
    MODEL_PATH,
    LABEL_COL,
    FEATURE_COLS,
)

# =========================
# 你可以把这两个路径先写死在这里
# 等后面想统一进 config.py 也可以再挪过去
# =========================
PLATT_PATH = "models/calibrated_platt.joblib"
ISOTONIC_PATH = "models/calibrated_isotonic.joblib"


def load_xy(csv_path, label_col, feature_cols=None):
    """
    读取 csv，并拆成 X / y

    参数：
    - csv_path: csv 文件路径
    - label_col: 标签列名，比如 target
    - feature_cols: 如果为 None，则默认取除标签外全部列作为特征

    返回：
    - X: 特征表
    - y: 标签数组
    """
    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {csv_path}")

    y = df[label_col].astype(int).values

    if feature_cols is None:
        X = df.drop(columns=[label_col])
    else:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        X = df[feature_cols]

    return X, y


def print_metrics(name, y_true, y_prob):
    """
    打印一组简单指标，方便你先看 validation 上的拟合情况
    注意：这里只是看 valid 上的表现，不是最终 test 结果
    """
    auc = roc_auc_score(y_true, y_prob)
    ll = log_loss(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    print(f"[{name}]")
    print(f"  AUC     = {auc:.6f}")
    print(f"  LogLoss = {ll:.6f}")
    print(f"  Brier   = {brier:.6f}")


def main():
    print("[INFO] Loading validation data...")
    X_valid, y_valid = load_xy(VALID_PATH, LABEL_COL, FEATURE_COLS)

    print("[INFO] Loading base model...")
    base_model = joblib.load(MODEL_PATH)

    # =========================
    # 先看看 raw model 在 valid 上的概率输出
    # 这只是为了做一个对比参考
    # =========================
    raw_prob = base_model.predict_proba(X_valid)[:, 1]
    print_metrics("RAW on VALID", y_valid, raw_prob)

    # =========================
    # 1) Platt scaling
    # method="sigmoid" 就是 Platt scaling
    # cv="prefit" 表示：底层模型已经训练好了，
    # 这里只在 valid 上拟合校准器，不重新训练底层模型
    # =========================
    print("[INFO] Fitting Platt scaling on validation set...")
    frozen_base_model = FrozenEstimator(base_model)
    platt_model = CalibratedClassifierCV(frozen_base_model, method="sigmoid")
    platt_model.fit(X_valid, y_valid)

    platt_prob = platt_model.predict_proba(X_valid)[:, 1]
    print_metrics("PLATT on VALID", y_valid, platt_prob)

    # 保存 platt 校准后的模型
    joblib.dump(platt_model, PLATT_PATH)
    print(f"[INFO] Saved Platt model -> {PLATT_PATH}")

    # =========================
    # 2) Isotonic regression
    # method="isotonic" 表示用保序回归做校准
    # 同样只在 valid 上拟合校准器
    # =========================
    print("[INFO] Fitting Isotonic regression on validation set...")
    frozen_base_model = FrozenEstimator(base_model)
    isotonic_model = CalibratedClassifierCV(frozen_base_model, method="isotonic")
    isotonic_model.fit(X_valid, y_valid)

    isotonic_prob = isotonic_model.predict_proba(X_valid)[:, 1]
    print_metrics("ISOTONIC on VALID", y_valid, isotonic_prob)

    # 保存 isotonic 校准后的模型
    joblib.dump(isotonic_model, ISOTONIC_PATH)
    print(f"[INFO] Saved Isotonic model -> {ISOTONIC_PATH}")

    print("[INFO] Calibration finished.")
    print("[INFO] Next step: evaluate raw / platt / isotonic on TEST set.")


if __name__ == "__main__":
    main()