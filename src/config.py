"""
Единая точка конфигурации проекта.
Все пути и гиперпараметры задаются здесь.
"""
from pathlib import Path

# Корень проекта
ROOT_DIR = Path(__file__).resolve().parent.parent

# Пути к данным
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Имена файлов
RAW_TRAIN_FILE = RAW_DIR / "train.csv"
TRAIN_FINAL_FILE = PROCESSED_DIR / "train_final.parquet"
TEST_FINAL_FILE = PROCESSED_DIR / "test_final.parquet"

# Жёсткие правила проекта (из ТЗ)
HOLD_OUT_SIZE = 1_000_000  # Последние 1 млн строк — это тест
TARGET_COL = "Response"
POSITIVE_CLASS = 1         # Купил страховку
MAIN_METRIC = "f1"         # Целевая метрика: F1-score для класса 1

# Гиперпараметры по умолчанию
RANDOM_SEED = 42
N_FOLDS = 5

# Итоговый список признаков (после Feature Engineering Саши)
FINAL_FEATURES = [
    'Age', 'Previously_Insured', 'Annual_Premium', 'Vintage',
    'Region_Code_target_enc', 'Policy_Sales_Channel_target_enc',
    'Prem_mean_Region_Code', 'Prem_std_Region_Code',
    'Prem_mean_Policy_Sales_Channel', 'Prem_std_Policy_Sales_Channel',
    'Prem_per_Age', 'Age_x_Vintage',
    'Vehicle_Age_enc', 'Vehicle_Damage_enc', 'Gender_enc'
]
