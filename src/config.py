from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_TRAIN_FILE = RAW_DIR / "train.csv"
TRAIN_FINAL_FILE = PROCESSED_DIR / "train_final.csv"
TEST_FINAL_FILE = PROCESSED_DIR / "test_final.csv"

HOLD_OUT_SIZE = 1_000_000
TARGET_COL = "Response"
POSITIVE_CLASS = 1
RANDOM_SEED = 42
N_FOLDS = 5
RARE_THRESHOLD = 0.01 # доля категории, ниже которой она уходит в 'Other'
CORR_THRESHOLD = 0.90 # порог мультиколлинеарности

# --- Категориальные признаки с высоким кардиналом ---
CAT_COLS = ["Region_Code", "Policy_Sales_Channel"]

# Финальный список из 15 признаков (Driving_License и id исключены)
FINAL_FEATURES = [
    'Age', 'Previously_Insured', 'Annual_Premium', 'Vintage',
    'Region_Code_target_enc', 'Policy_Sales_Channel_target_enc',
    'Prem_mean_Region_Code', 'Prem_std_Region_Code',
    'Prem_mean_Policy_Sales_Channel', 'Prem_std_Policy_Sales_Channel',
    'Prem_per_Age', 'Age_x_Vintage',
    'Vehicle_Age_enc', 'Vehicle_Damage_enc', 'Gender_enc'
]

# --- Варианты обработанных датасетов (parquet в PROCESSED_DATA_DIR) ---
VARIANTS = {
    "v1_te":        {"train": "v1_te.parquet",        "test": "v1_te_test.parquet",        "cat_features": []},
    "v2_ohe":       {"train": "v2_ohe.parquet",       "test": "v2_ohe_test.parquet",       "cat_features": []},
    "v3_freq":      {"train": "v3_freq.parquet",      "test": "v3_freq_test.parquet",      "cat_features": []},
    "v4_woe":       {"train": "v4_woe.parquet",       "test": "v4_woe_test.parquet",       "cat_features": []},
    "v5_benchmark": {"train": "v5_benchmark.parquet", "test": "v5_benchmark_test.parquet",
                     "cat_features": list(CAT_COLS)},
}