"""Feature Engineering: кодирование категорий, агрегации, взаимодействия, корреляции."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold

from src.config import (CAT_COLS, CORR_THRESHOLD, N_FOLDS, RANDOM_SEED,
                        RARE_THRESHOLD, TARGET_COL)


def target_encode_kfold(train_df, test_df, cat_cols=None, target_col=TARGET_COL, smoothing=10):
    """Target Encoding с Bayesian smoothing и защитой от leakage (KFold по train)."""
    if cat_cols is None:
        cat_cols = CAT_COLS
    global_mean = train_df[target_col].mean()

    for col in cat_cols:
        enc_col = f"{col}_target_enc"
        train_df[enc_col] = np.nan
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        for train_idx, val_idx in skf.split(train_df, train_df[target_col]):
            fold = train_df.iloc[train_idx]
            means = fold.groupby(col, observed=True)[target_col].mean()
            counts = fold.groupby(col, observed=True)[target_col].count()
            smoothed = (counts * means + smoothing * global_mean) / (counts + smoothing)
            train_df.loc[train_df.index[val_idx], enc_col] = (
                train_df.iloc[val_idx][col].map(smoothed).fillna(global_mean))
        # test кодируем статистиками со всего train
        means = train_df.groupby(col, observed=True)[target_col].mean()
        counts = train_df.groupby(col, observed=True)[target_col].count()
        smoothed = (counts * means + smoothing * global_mean) / (counts + smoothing)
        test_df[enc_col] = test_df[col].map(smoothed).fillna(global_mean)
    return train_df, test_df


def frequency_encode(train_df, test_df, cat_cols=None):
    """Заменяет категорию на частоту её встречаемости (fit на train)."""
    if cat_cols is None:
        cat_cols = CAT_COLS
    for col in cat_cols:
        freq = train_df[col].value_counts(normalize=True)
        train_df[f"{col}_freq"] = train_df[col].map(freq)
        test_df[f"{col}_freq"] = test_df[col].map(freq).fillna(freq.mean())
    return train_df, test_df


def woe_encode(train_df, test_df, cat_cols=None, target_col=TARGET_COL):
    """Weight of Evidence: ln(P(0|cat)/P(1|cat)) с epsilon-сглаживанием (fit на train)."""
    if cat_cols is None:
        cat_cols = CAT_COLS
    eps = 1e-6
    for col in cat_cols:
        cross = pd.crosstab(train_df[col], train_df[target_col])
        pos_rate = (cross[1] + eps) / (cross[1].sum() + eps)
        neg_rate = (cross[0] + eps) / (cross[0].sum() + eps)
        woe = np.log(neg_rate / pos_rate)
        train_df[f"{col}_woe"] = train_df[col].map(woe)
        test_df[f"{col}_woe"] = test_df[col].map(woe).fillna(0.0)
    return train_df, test_df


def encode_categoricals(train_df, test_df):
    """Ручное кодирование: Vehicle_Age (ordinal 0/1/2), Vehicle_Damage и Gender (binary)."""
    age_mapping = {"< 1 Year": 0, "1-2 Year": 1, "> 2 Years": 2}
    for df in (train_df, test_df):
        df["Vehicle_Age_enc"] = df["Vehicle_Age"].map(age_mapping).astype("int8")
        df["Vehicle_Damage_enc"] = (df["Vehicle_Damage"] == "Yes").astype("int8")
        df["Gender_enc"] = (df["Gender"] == "Male").astype("int8")
    return train_df, test_df


def group_rare_categories(df, col, threshold=RARE_THRESHOLD, rare_cats=None):
    """Группировка редких категорий (< threshold) в 'Other'. Возвращает (df, rare_cats)."""
    if rare_cats is None:
        counts = df[col].value_counts(normalize=True)
        rare_cats = counts[counts < threshold].index.tolist()
    df[col] = df[col].astype(object).where(~df[col].isin(rare_cats), "Other")
    return df, rare_cats


def create_aggregations(train_df, test_df, group_cols=None):
    """Среднее и std Annual_Premium внутри категорий (контекст «выше/ниже своего региона»)."""
    if group_cols is None:
        group_cols = CAT_COLS
    for df in (train_df, test_df):
        for col in group_cols:
            df[f"Prem_mean_{col}"] = df.groupby(col, observed=True)["Annual_Premium"].transform("mean")
            df[f"Prem_std_{col}"] = df.groupby(col, observed=True)["Annual_Premium"].transform("std").fillna(0)
    return train_df, test_df


def create_interactions(train_df, test_df):
    """Взаимодействия: Prem_per_Age (взнос на единицу возраста), Age_x_Vintage (зрелость)."""
    for df in (train_df, test_df):
        df["Prem_per_Age"] = df["Annual_Premium"] / (df["Age"] + 1)
        df["Age_x_Vintage"] = df["Age"] * df["Vintage"]
    return train_df, test_df


def analyze_correlations(train, target=TARGET_COL, sample=500_000,
                         corr_threshold=CORR_THRESHOLD, show_plot=True):
    """Корреляционный анализ (ТЗ п.5): heatmap, мультиколлинеарность, связь с таргетом.
    НИЧЕГО не сохраняет на диск — только вывод и возвращаемый словарь."""
    df = train.sample(n=min(sample, len(train)), random_state=RANDOM_SEED)
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[num].corr()

    if show_plot:
        size = max(8, len(num) * 0.4)
        plt.figure(figsize=(size, size))
        sns.heatmap(corr, cmap="coolwarm", center=0, square=True, linewidths=.3)
        plt.title("Correlation matrix")
        plt.tight_layout()
        plt.show()

    cols = [c for c in num if c != target]
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) >= corr_threshold:
                pairs.append({"feature_1": a, "feature_2": b, "corr": round(r, 4)})
    pairs_df = (pd.DataFrame(pairs, columns=["feature_1", "feature_2", "corr"])
                .sort_values("corr", key=abs, ascending=False)
                if pairs else pd.DataFrame(columns=["feature_1", "feature_2", "corr"]))

    target_corr = (corr[target].drop(target).sort_values(key=abs, ascending=False)
                   if target in corr.columns else pd.Series(dtype=float))

    print(f"Пар с |r| >= {corr_threshold}: {len(pairs_df)}")
    if len(pairs_df):
        print(pairs_df.to_string(index=False))
    print("\nТоп-10 корреляций с таргетом:")
    print(target_corr.head(10).to_string())

    return {"corr_matrix": corr, "multicollinear_pairs": pairs_df, "target_corr": target_corr}