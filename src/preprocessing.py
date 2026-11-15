"""Загрузка и предобработка: сплит, downcast, варианты датасетов, масштабирование в памяти."""
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.config import (HOLD_OUT_SIZE, PROCESSED_DATA_DIR, RAW_TRAIN_FILE,
                        TARGET_COL, VARIANTS)


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
    """Читает CSV, даункастит, отрезает последние hold_out_size строк в hold-out test."""
    df = pd.read_csv(path)
    df = safe_downcast(df)

    train_df = df.iloc[:-hold_out_size].copy().reset_index(drop=True)
    test_df = df.iloc[-hold_out_size:].copy().reset_index(drop=True)
    del df

    for frame in (train_df, test_df):
        if "id" in frame.columns:
            frame.drop("id", axis=1, inplace=True)

    test_target = test_df[TARGET_COL].copy()
    test_df = test_df.drop(TARGET_COL, axis=1)
    return train_df, test_df, test_target


def load_variant(name: str, data_dir: Path = PROCESSED_DATA_DIR):
    """Загрузка готового варианта датасета: (train, test, features, cat_features)."""
    cfg = VARIANTS[name]
    train = pd.read_parquet(Path(data_dir) / cfg["train"])
    test = pd.read_parquet(Path(data_dir) / cfg["test"])
    cat = list(cfg["cat_features"])
    if cat:  # CatBoost требует str/category для нативных категорий
        train[cat] = train[cat].astype(str)
        test[cat] = test[cat].astype(str)
    features = [c for c in train.columns if c != TARGET_COL]
    return train, test, features, cat


def combine_variants(parts: list, data_dir: Path = PROCESSED_DATA_DIR):
    """Сборка нового датасета из колонок разных вариантов (строки выровнены).

    parts = [("v5_benchmark", None), ("v1_te", ["Region_Code_target_enc"])]
    None — взять все признаки варианта; Response берётся из первого варианта.
    """
    tr_out, te_out, n = None, None, None
    for name, cols in parts:
        tr, te, feats, _ = load_variant(name, data_dir)
        if cols is None:
            cols = feats
        if tr_out is None:
            n = len(tr)
            tr_out = tr[cols + [TARGET_COL]].copy()
            te_out = te[cols].copy()
        else:
            if len(tr) != n:
                raise ValueError("Строки вариантов не выровнены!")
            new = [c for c in cols if c not in tr_out.columns]
            tr_out = pd.concat([tr_out, tr[new].reset_index(drop=True)], axis=1)
            te_out = pd.concat([te_out, te[new].reset_index(drop=True)], axis=1)
    return tr_out, te_out


def scale_features(train_df, test_df, features, cat_features=None, method="standard"):
    """Масштабирование числовых признаков полностью в памяти, ничего не пишется на диск.

    Скалер обучается ТОЛЬКО на train и сразу применяется к train и test.
    method: "standard" (Z-score) или "minmax" ([0, 1]).
    Возвращает (train_scaled, test_scaled, scaler) — scaler остаётся в памяти,
    при необходимости им можно.transform() новые данные.
    Для CatBoost масштабирование не нужно; для LogReg / RF / SVM — обязательно.
    """
    if method not in ("standard", "minmax"):
        raise ValueError("method должен быть 'standard' или 'minmax'")

    cat_features = cat_features or []
    num = [c for c in features if c not in cat_features]

    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    scaler.fit(train_df[num])

    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    train_scaled[num] = scaler.transform(train_df[num])
    test_scaled[num] = scaler.transform(test_df[num])
    return train_scaled, test_scaled, scaler