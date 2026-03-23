# config.py
# 这个文件统一管理项目中的公共配置：
# 1. 数据路径
# 2. 模型路径
# 3. 输出路径
# 4. 标签列名
# 5. 实验参数

from pathlib import Path

# =========================
# 1) 目录路径
# =========================

# 当前 config.py 所在目录：SRC
SRC_DIR = Path(__file__).resolve().parent

# 项目根目录：SRC 的上一层
BASE_DIR = SRC_DIR.parent

# 数据目录
DATA_DIR = BASE_DIR / "data"

# 模型目录
MODEL_DIR = BASE_DIR / "models"

# 输出目录
OUTPUT_DIR = BASE_DIR / "outputs"

# 图片输出目录
FIGURE_DIR = OUTPUT_DIR / "figures"

# 自动创建输出文件夹，避免保存时报错
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 2) 数据文件路径
# =========================

# 原始总数据
RAW_DATA_PATH = DATA_DIR / "Criteo_1M_with_nans.csv"

# 切分后的训练 / 验证 / 测试集
TRAIN_PATH = DATA_DIR / "train.csv"
VALID_PATH = DATA_DIR / "valid.csv"
TEST_PATH = DATA_DIR / "test.csv"

# =========================
# 3) 模型文件路径
# =========================

# 原始 LR pipeline 模型
MODEL_PATH = MODEL_DIR / "lr_pipeline.joblib"

# 校准后的模型
PLATT_PATH = MODEL_DIR / "calibrated_platt.joblib"
ISOTONIC_PATH = MODEL_DIR / "calibrated_isotonic.joblib"

# =========================
# 4) 输出文件路径
# =========================

# 默认指标文件（旧版脚本可能会用到）
METRICS_PATH = OUTPUT_DIR / "metrics.json"

# =========================
# 5) 数据配置
# =========================

# 标签列名
LABEL_COL = "target"

# 特征列配置
# None 表示：默认使用“除标签列外的所有列”
FEATURE_COLS = None

# =========================
# 6) 实验参数
# =========================

# 随机种子
RANDOM_STATE = 42

# ECE / reliability diagram 的分桶数
N_BINS = 15