"""Загрузка и комбинирование вариантов датасетов (Data Engineering stage)."""
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data_versions")

VARIANTS = {
    "v1_te":        {"train": "v1_te.parquet",        "test": "v1_te_test.parquet",        "cat_features": []},
    "v2_ohe":       {"train": "v2_ohe.parquet",       "test": "v2_ohe_test.parquet",       "cat_features": []},
    "v3_freq":      {"train": "v3_freq.parquet",      "test": "v3_freq_test.parquet",      "cat_features": []},
    "v4_woe":       {"train": "v4_woe.parquet",       "test": "v4_woe_test.parquet",       "cat_features": []},
    "v5_benchmark": {"train": "v5_benchmark.parquet", "test": "v5_benchmark_test.parquet",
                     "cat_features": ["Region_Code", "Policy_Sales_Channel"]},
}


def load_variant(name, data_dir=DATA_DIR):
    """Возвращает (train, test, features, cat_features)."""
    cfg = VARIANTS[name]
    train = pd.read_parquet(Path(data_dir) / cfg["train"])
    test = pd.read_parquet(Path(data_dir) / cfg["test"])
    cat = cfg["cat_features"]
    if cat:  # CatBoost требует str для нативных категорий
        train[cat] = train[cat].astype(str)
        test[cat] = test[cat].astype(str)
    features = [c for c in train.columns if c != "Response"]
    return train, test, features, cat


def combine_variants(parts, data_dir=DATA_DIR):
    """Собирает новый датасет из базисных вариантов: parts = [("v5_benchmark", None), ("v1_te", ["Region_Code_target_enc"])]
    None = все признаки; Response берётся из первого варианта.
    """
    tr_out, te_out, n = None, None, None
    for name, cols in parts:
        tr, te, feats, _ = load_variant(name, data_dir)
        if cols is None:
            cols = feats
        if tr_out is None:
            n = len(tr)
            tr_out = tr[cols + ["Response"]].copy()
            te_out = te[cols].copy()
        else:
            assert len(tr) == n, "Строки не выровнены!"
            new = [c for c in cols if c not in tr_out.columns]
            tr_out = pd.concat([tr_out, tr[new].reset_index(drop=True)], axis=1)
            te_out = pd.concat([te_out, te[new].reset_index(drop=True)], axis=1)
    return tr_out, te_out
