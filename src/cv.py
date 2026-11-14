import pandas as pd
from sklearn.model_selection import StratifiedKFold

def get_cv_folds(df, target_col='Response', n_splits=5, random_state=42):
    """
    Разбивает датасет на фолды для кросс-валидации строго в оперативной памяти.
    Возвращает список словарей вида: {'fold_id': int, 'train': DataFrame, 'valid': DataFrame}
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    for fold_num, (train_idx, val_idx) in enumerate(skf.split(df, df[target_col])):
        df_train = df.iloc[train_idx].copy()
        df_val = df.iloc[val_idx].copy()        
        folds.append({'fold_id': fold_num, 'train': df_train, 'valid': df_val})
        print(f"Fold {fold_num}: Train = {len(df_train):,} строк | Valid = {len(df_val):,} строк")
        
    print("Фолды готовы. Используйте цикл `for fold in folds:` для обучения.")
    return folds
