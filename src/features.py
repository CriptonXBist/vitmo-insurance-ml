"""
Feature Engineering: Target Encoding, агрегации, кодирование.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.config import N_FOLDS, RANDOM_SEED, TARGET_COL

def target_encode_kfold(train_df: pd.DataFrame, test_df: pd.DataFrame, cat_cols: list, smoothing: int = 10):
    """Target Encoding с Bayesian smoothing и защитой от leakage через K-Fold."""
    global_mean = train_df[TARGET_COL].mean()
    
    for col in cat_cols:
        train_df[f'{col}_target_enc'] = np.nan
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        
        for train_idx, val_idx in skf.split(train_df, train_df[TARGET_COL]):
            fold = train_df.iloc[train_idx]
            means = fold.groupby(col, observed=True)[TARGET_COL].mean()
            counts = fold.groupby(col, observed=True)[TARGET_COL].count()
            smoothed = (counts * means + smoothing * global_mean) / (counts + smoothing)
            
            train_df.loc[train_df.index[val_idx], f'{col}_target_enc'] = (
                train_df.iloc[val_idx][col].map(smoothed).fillna(global_mean)
            )
            
        # Для теста используем среднее по всему трейну
        means = train_df.groupby(col, observed=True)[TARGET_COL].mean()
        counts = train_df.groupby(col, observed=True)[TARGET_COL].count()
        smoothed = (counts * means + smoothing * global_mean) / (counts + smoothing)
        test_df[f'{col}_target_enc'] = test_df[col].map(smoothed).fillna(global_mean)
        
    return train_df, test_df

def encode_categoricals(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Ручное кодирование: порядковое и бинарное."""
    age_mapping = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
    
    for df in (train_df, test_df):
        df['Vehicle_Age_enc'] = df['Vehicle_Age'].map(age_mapping).astype(int)
        df['Vehicle_Damage_enc'] = (df['Vehicle_Damage'] == 'Yes').astype(int)
        df['Gender_enc'] = (df['Gender'] == 'Male').astype(int)
        
    return train_df, test_df

def group_rare_categories(df: pd.DataFrame, col: str, threshold: float = 0.01, rare_cats: list = None):
    """Группировка редких категорий в 'Other'."""
    if rare_cats is None:
        counts = df[col].value_counts(normalize=True)
        rare_cats = counts[counts < threshold].index.tolist()
    
    df[col] = df[col].astype(object).where(~df[col].isin(rare_cats), "Other")
    return df, rare_cats
