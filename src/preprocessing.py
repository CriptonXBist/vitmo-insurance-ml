"""
Модуль для загрузки и первичной предобработки данных.
"""
import pandas as pd
import numpy as np
from src.config import HOLD_OUT_SIZE, RAW_TRAIN_FILE, TARGET_COL

def safe_downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Оптимизация памяти: понижение разрядности числовых типов."""
    for col in df.select_dtypes(include=["integer"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["floating"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["object", "string"]).columns:
        if df[col].nunique() < 50:
            df[col] = df[col].astype("category")
    return df

def load_and_split(path: str = str(RAW_TRAIN_FILE), hold_out_size: int = HOLD_OUT_SIZE):
    """
    Загружает данные и строго разделяет их: 
    последние hold_out_size строк идут в тест (без Response).
    """
    df = pd.read_csv(path)
    df = safe_downcast(df)
    
    # Строгий сплит по порядку строк (без random_state!)
    train_df = df.iloc[:-hold_out_size].copy().reset_index(drop=True)
    test_df = df.iloc[-hold_out_size:].copy().reset_index(drop=True)
    
    # Очистка от служебных колонок
    for frame in (train_df, test_df):
        if "id" in frame.columns:
            frame.drop("id", axis=1, inplace=True)
            
    # Сохраняем таргет теста отдельно для финальной оценки
    test_target = test_df[TARGET_COL].copy()
    test_df = test_df.drop(TARGET_COL, axis=1)
    
    return train_df, test_df, test_target
