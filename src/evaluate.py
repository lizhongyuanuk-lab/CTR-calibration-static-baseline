# evaluate.py
# 作用：
# 1. 读取测试集
# 2. 根据 model_type 读取不同模型（raw / platt / isotonic）
# 3. 在测试集上输出点击概率
# 4. 计算评估指标（AUC / LogLoss / Brier / ECE）
# 5. 保存指标到 json 文件（带时间戳，不覆盖历史结果）
# 6. 画图（可靠性图、概率分布图、bin gap 图，带时间戳）
# 7. 支持 run_tag，方便区分不同实验轮次

import json
import argparse
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

from config import (
    TEST_PATH,
    MODEL_PATH,
    PLATT_PATH,
    ISOTONIC_PATH,
    LABEL_COL,
    FEATURE_COLS,
    N_BINS,
    FIGURE_DIR,
    OUTPUT_DIR,
)

from plots import (
    plot_reliability_diagram,
    plot_prediction_histogram,
    plot_bin_gap
)


def load_xy(csv_path, label_col, feature_cols=None):
    """
    读取 csv，并拆成 X / y
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


def expected_calibration_error(y_true, y_prob, n_bins=15):
    """
    计算 ECE（Expected Calibration Error）
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    ece = 0.0
    total = len(y_true)

    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue

        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / total) * abs(acc - conf)

    return float(ece)


def evaluate_model(model, X, y, n_bins=15):
    """
    计算各类评估指标
    """
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "auc": float(roc_auc_score(y, y_prob)),
        "logloss": float(log_loss(y, y_prob)),
        "brier": float(brier_score_loss(y, y_prob)),
        "ece": float(expected_calibration_error(y, y_prob, n_bins=n_bins)),
        "avg_pred": float(np.mean(y_prob)),
        "true_ctr": float(np.mean(y)),
    }

    return metrics, y_prob


def get_model_path(model_type: str):
    """
    根据 model_type 返回对应模型路径
    """
    if model_type == "raw":
        return MODEL_PATH
    elif model_type == "platt":
        return PLATT_PATH
    elif model_type == "isotonic":
        return ISOTONIC_PATH
    else:
        raise ValueError("model_type must be one of: raw, platt, isotonic")


def build_output_stem(model_type: str, run_tag: str | None):
    """
    构造不重复的输出文件名主干
    例如：
    raw_20260309_120530
    platt_after_calibration_20260309_120530
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if run_tag is not None and run_tag.strip() != "":
        safe_tag = run_tag.strip().replace(" ", "_")
        stem = f"{model_type}_{safe_tag}_{timestamp}"
    else:
        stem = f"{model_type}_{timestamp}"

    return stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default="raw",
        choices=["raw", "platt", "isotonic"],
        help="Choose which model to evaluate"
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="Optional tag to distinguish different runs, e.g. baseline / after_calibration / v2"
    )
    args = parser.parse_args()

    model_type = args.model_type
    run_tag = args.run_tag

    print("[INFO] Loading test data...")
    X_test, y_test = load_xy(TEST_PATH, LABEL_COL, FEATURE_COLS)

    model_path = get_model_path(model_type)
    print(f"[INFO] Loading model: {model_type} -> {model_path}")
    model = joblib.load(model_path)

    print("[INFO] Evaluating...")
    metrics, y_prob = evaluate_model(model, X_test, y_test, n_bins=N_BINS)

    print(f"[RESULT] ({model_type})")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    # 构造不会覆盖的文件名主干
    stem = build_output_stem(model_type, run_tag)

    # 保存 metrics json
    metrics_path = OUTPUT_DIR / f"metrics_{stem}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 保存图片
    reliability_path = FIGURE_DIR / f"reliability_{stem}.png"
    pred_hist_path = FIGURE_DIR / f"pred_hist_{stem}.png"
    bin_gap_path = FIGURE_DIR / f"bin_gap_{stem}.png"

    title_suffix = f"{model_type.upper()}"
    if run_tag.strip():
        title_suffix += f" | {run_tag}"
    title_suffix += " | Test Set"

    plot_reliability_diagram(
        y_test,
        y_prob,
        reliability_path,
        n_bins=N_BINS,
        title=f"{title_suffix} | Reliability Diagram"
    )

    plot_prediction_histogram(
        y_prob,
        pred_hist_path,
        title=f"{title_suffix} | Predicted Probability Histogram"
    )

    plot_bin_gap(
        y_test,
        y_prob,
        bin_gap_path,
        n_bins=N_BINS,
        title=f"{title_suffix} | Calibration Gap by Bin"
    )

    print(f"[INFO] model_type = {model_type}")
    print(f"[INFO] run_tag    = {run_tag if run_tag.strip() else '(none)'}")
    print(f"[INFO] Metrics saved to: {metrics_path}")
    print(f"[INFO] Reliability figure saved to: {reliability_path}")
    print(f"[INFO] Histogram figure saved to: {pred_hist_path}")
    print(f"[INFO] Bin-gap figure saved to: {bin_gap_path}")


if __name__ == "__main__":
    main()