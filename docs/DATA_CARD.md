# этап Data Engineering

## 1. Описание признаков
Целевая: `Response` (1 = купил автостраховку, ~12.3%). Train: 10 504 798 строк, test_locked: последние 1 000 000.

### Исходные
| Признак | Смысл |
|---|---|
| Age | Возраст клиента |
| Driving_License | Наличие прав (99.8% = 1) |
| Region_Code | Код региона (0 — валидный сегмент, не пропуск) |
| Previously_Insured | Есть ли уже автостраховка |
| Annual_Premium | Годовой взнос (2 630 – 540 165) |
| Vintage | Сколько дней обслуживается в компании |
| Vehicle_Age | Возраст авто / повреждения / пол |
| Vehicle_Damage | факт повреждения (1 - была авария) |
| Gender | пол |

### Созданные (Feature Engineering)
| Признак | Метод |
|---|---|
| Vehicle_Age_enc (0/1/2), Vehicle_Damage_enc, Gender_enc | Ordinal / binary (= One hot drop_first для бинарных) |
| *_target_enc | Target Encoding + Bayesian smoothing (s=10) |
| *_freq | Frequency Encoding |
| *_woe | WOE = ln(neg_rate/pos_rate) |
| Prem_mean_{col}, Prem_std_{col} | Агрегации взноса по региону/каналу |
| Prem_per_Age | Annual_Premium / (Age+1) |
| Age_x_Vintage | Age × Vintage |

## 2. Пропуски и аномалии
Изначальных NaN нет; нули — валидные состояния. id удалён. Region_Code=0 оставлен (конверсия 7.5% vs 12.3%). NaN после FE: std→0, mean/enc→медиана/глобальное среднее.

## 3. Варианты датасетов (data_versions/, parquet; строки выровнены)
| Вариант | Кодирование Region/Channel | Признаков | F1 test_locked |
|---|---|---|---|
| v5_benchmark | нативный CatBoost (cat_features) | 10 | **0.4866** |
| v1_te | Target Encoding | 18 | 0.4856 |
| v4_woe | WOE | 16 | 0.4855 |
| v3_freq | Frequency | 16 | 0.4826 |
| v2_ohe | OHE (rare→Other, drop_first) | 45 | 0.4823 |

## 4. Как использовать
- Загрузка: `from src.data_utils import load_variant, combine_variants`
- Валидация: 5-fold StratifiedKFold, фолды зафиксированы: `src.cv.load_folds()` (artifacts/folds/folds_5.npz)
- Скалеры (для LogReg/RF): `src.scaling.apply_scaler` (обучены на train; Standard + MinMax)
- Порог классификации ≈ 0.70 (подбор на валидации дал +0.035 F1)
- CatBoost масштабирование НЕ нужно (снижает F1)

## 5. Отчёты
reports/: heatmap + multicollinearity_*.csv + target_corr_*.csv для всех 5 вариантов.
Мультиколлинеарность |r|≥0.9: только 1 пара в v2_ohe, в остальных 0.

## 6. Варианты датасетов 
- Папка: data_versions/
- Скачать архив: https://disk.yandex.ru/d/Z5rxxwMp43ZYMw
- Распаковать в корень репозитория, переместить в папку data_versions/
- Состав: v1_te / v2_ohe / v3_freq / v4_woe / v5_benchmark (+ *_test) и test_locked.parquet.
- Способы работы с датасетами задаются функциями в файле src.data_utils
