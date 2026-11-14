import pandas as pd
import numpy as np

def analyze_correlations(df, target_col='Response', threshold=0.85):
    """
    Возвращает словарь:
    - 'corr_matrix': полная матрица корреляций
    - 'target_corr': корреляция признаков с таргетом
    - 'multicollinear_pairs': список пар мультиколлинеарных признаков
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    if target_col not in num_cols:
        raise ValueError(f"Таргет '{target_col}' не найден среди числовых признаков.")        
    corr_matrix = df[num_cols].corr()
    
    # 1. Связь с таргетом
    target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
    print(f"\nТоп-10 признаков по связи с '{target_col}':")
    for feat, val in target_corr.head(10).items():
        print(f"   • {feat}: {val:.4f}")
    
    # 2. Поиск мультиколлинеарности (берем верхнюю треугольную матрицу)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    multicollinear_pairs = []
    for column in upper.columns:
        highly_correlated = upper.index[upper[column].abs() > threshold].tolist()
        for row in highly_correlated:
            multicollinear_pairs.append((row, column, upper.loc[row, column]))
            
    print(f"\nНайдено мультиколлинеарных пар: {len(multicollinear_pairs)}")
    if multicollinear_pairs:
        for feat1, feat2, corr_val in multicollinear_pairs[:5]:
            print(f"{feat1} <-> {feat2}: {corr_val:.4f}")
        if len(multicollinear_pairs) > 5:
            print(f"... и еще {len(multicollinear_pairs) - 5} пар.")
            
    # 3. Определяем, что удалить (удаляем тот признак из пары, который слабее связан с таргетом)
    features_to_drop = set()
    for feat1, feat2, _ in multicollinear_pairs:
        if feat1 in target_corr.index and feat2 in target_corr.index:
            if abs(target_corr[feat1]) < abs(target_corr[feat2]):
                features_to_drop.add(feat1)
            else:
                features_to_drop.add(feat2)
                
    print(f"\nК удалению (для устранения мультиколлинеарности): {list(features_to_drop)}")
    
    return {
        "corr_matrix": corr_matrix,
        "target_corr": target_corr,
        "multicollinear_pairs": multicollinear_pairs,
        "features_to_drop": list(features_to_drop)
    }
