"""Масштабирование. Скалеры ТОЛЬКО для train, хранятся как артефакты.
Для CatBoost не нужно; для LogReg/RF/SVM — обязательно."""
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler

ART_DIR = Path("artifacts/scalers")


def fit_scalers(train, features, cat_features, variant, out_dir=ART_DIR):
    num = [c for c in features if c not in cat_features]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for method, cls in [("standard", StandardScaler), ("minmax", MinMaxScaler)]:
        sc = cls()
        sc.fit(train[num])
        joblib.dump({"scaler": sc, "num_features": num}, out / f"{variant}_{method}.joblib")
    return num


def apply_scaler(df, variant, method="standard", art_dir=ART_DIR):
    bundle = joblib.load(Path(art_dir) / f"{variant}_{method}.joblib")
    out = df.copy()
    num = bundle["num_features"]
    out[num] = bundle["scaler"].transform(out[num])
    return out
