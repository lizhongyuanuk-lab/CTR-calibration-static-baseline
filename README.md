CTR-calibration-static-baseline

A reproducible static CTR calibration baseline using logistic regression, Platt scaling, and isotonic regression.

Overview

This project studies post-hoc probability calibration in a static click-through rate (CTR) prediction setting.
The main goal is not to improve ranking performance itself, but to examine whether probability estimates can be made more reliable while ranking performance remains broadly stable.

The repository includes:

a logistic regression baseline for static CTR prediction
two standard post-hoc calibration methods:
Platt scaling
isotonic regression
evaluation code for:
ROC-AUC
LogLoss
Brier score
Expected Calibration Error (ECE)
reliability diagrams
calibration-gap-by-bin plots
predicted-probability histograms
Motivation

In industrial advertising systems, CTR prediction is important not only because it affects ranking, but also because predicted probabilities may be used in downstream estimation and decision-making. In that setting, a practical question is whether probability calibration can be improved without materially disturbing ranking performance.

This repository provides a simple and reproducible static baseline for that question.

Project structure
CTR-calibration-static-baseline/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── models/
│   └── README.md
├── outputs/
│   └── README.md
└── src/
    ├── calibrate.py
    ├── config.py
    ├── evaluate.py
    ├── peek_data.py
    ├── plots.py
    ├── train.py
    └── utils.py
Dataset

This repository does not redistribute the full dataset.

The project uses a static tabular CTR dataset in a Criteo-style format, with:

one binary target column: target
numerical features such as intCol_*
categorical features such as catCol_*
missing values handled through preprocessing
Data source

A suitable public source for this project is the Criteo Display Advertising Challenge dataset on Kaggle. The challenge page states that Criteo shared one week of anonymised data for CTR prediction, and the corresponding Kaggle data page describes the training set as a portion of Criteo traffic over 7 days. The same dataset is also listed on the Criteo AI Lab resources page as the Kaggle Display Advertising dataset.

Expected raw data file

Place the raw dataset under:

data/Criteo_1M_with_nans.csv
Experimental pipeline

The workflow consists of three stages:

Train a raw logistic regression baseline on the training split
Fit post-hoc calibration models on the validation split
Evaluate all models on the held-out test split

The calibration methods evaluated are:

Platt scaling
isotonic regression
Installation

Create a Python environment and install dependencies:

pip install -r requirements.txt
Reproducibility

To reproduce the full static experiment, run:

python src/train.py
python src/calibrate.py
python src/evaluate.py --model_type raw --run_tag baseline
python src/evaluate.py --model_type platt --run_tag calibrated
python src/evaluate.py --model_type isotonic --run_tag calibrated
Data splitting

The training script generates stratified splits with the following final ratio:

60% training
20% validation
20% test

The validation split is used only for calibration fitting, while the test split is reserved for final evaluation.

Models
Raw baseline

A logistic regression pipeline with:

median imputation for numerical features
standardisation for numerical features
most-frequent imputation for categorical features
one-hot encoding for categorical features
Calibrated models

Two post-hoc calibration methods are applied to the frozen baseline model:

Platt scaling (sigmoid)
isotonic regression
Evaluation metrics

This project evaluates both ranking quality and probability quality.

Ranking-oriented metric
ROC-AUC
Probability-oriented metrics
LogLoss
Brier score
ECE
reliability diagrams
calibration-gap-by-bin plots
predicted-probability histograms
Main finding

In this static setting, post-hoc calibration improves probability-oriented metrics substantially, while ROC-AUC changes very little. This suggests that, when ranking performance is expected to remain broadly stable, the practical effect of standard post-hoc calibration is reflected more clearly in probability-quality metrics than in ROC-AUC alone.

Important note

This repository is intended as a static baseline study rather than a complete industrial deployment framework.

Its main purpose is to provide:

a clean calibration baseline
a reproducible evaluation pipeline
a foundation for future work on temporal drift and non-stationary CTR prediction
Repository note

For a lightweight and reproducible repository, the following are not included in version control:

raw dataset files
train / validation / test csv files
trained .joblib model files
generated metrics files
generated figures

These artifacts can be reproduced locally by following the documented workflow above.
