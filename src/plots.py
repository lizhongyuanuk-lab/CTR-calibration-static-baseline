# plots.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def plot_reliability_diagram(y_true, y_prob, save_path, n_bins=15, title="Reliability Diagram"):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_prediction_histogram(y_prob, save_path, title="Predicted Probability Histogram"):
    plt.figure(figsize=(7, 4))
    plt.hist(y_prob, bins=50)
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_bin_gap(y_true, y_prob, save_path, n_bins=15, title="Calibration Gap by Bin"):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    xs, confs, accs = [], [], []

    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        xs.append(b)
        confs.append(conf)
        accs.append(acc)

    plt.figure(figsize=(8, 4))
    plt.plot(xs, confs, marker="o", label="Confidence")
    plt.plot(xs, accs, marker="o", label="Observed CTR")
    plt.xlabel("Bin index")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()