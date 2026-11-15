# Data Card — Insurance Cross-Sell

## 1. Описание признаков

Целевая переменная: `Response` (1 = купил автостраховку). Доля класса 1 ≈ 12.3% (дисбаланс 1:7).
Train: 10 504 798 строк. Hold-out (`test_locked`): последние 1 000 000 строк.

### Исходные
| Признак | Смысл |
|---|---|
| Age | Возраст клиента |
| Driving_License | Наличие прав (99.8% = 1) |
| Region_Code | Код региона (0 — валидный сегмент, не пропуск) |
| Policy_Sales_Channel | Код канала продаж (~150 уникальных) |
| Previously_Insured | Есть ли уже автостраховка |
| Annual_Premium | Годовой взнос (2 630 – 540 165) |
| Vintage | Сколько дней обслуживается в компании |
| Vehicle_Age | Возраст авто: <1 / 1–2 / >2 лет |
| Vehicle_Damage | Факт повреждения (Yes/No) |
| Gender | Пол |

### Созданные (Feature Engineering)
| Признак | Метод |
|---|---|
| Vehicle_Age_enc (0/1/2), Vehicle_Damage_enc, Gender_enc | Ordinal / binary (для бинарных = OHE drop_first) |
| *_target_enc | Target Encoding + Bayesian smoothing (s=10), KFold против leakage |
| *_freq | Frequency Encoding (частота категории, fit на train) |
| *_woe | WOE = ln(neg_rate / pos_rate), epsilon-сглаживание |
| Prem_mean_{col}, Prem_std_{col} | Агрегации взноса по региону / каналу |
| Prem_per_Age | Annual_Premium / (Age + 1) |
| Age_x_Vintage | Age × Vintage |

`id` удалён как служебный. `Driving_License` оставлен в вариантах по решению команды
(в `config.FINAL_FEATURES` его нет — это отдельный курируемый список из 15 признаков).

## 2. Пропуски и аномалии
Изначальных NaN нет — нули в бинарных признаках являются валидными состояниями («нет прав»,
«не застрахован»), а не пропусками. `Region_Code = 0` оставлен: конверсия 7.5% против 12.3%
в среднем, то есть это отдельный сегмент.
NaN возникают только после FE: `std` для группы из одного элемента → 0,
`mean` / `enc` для незнакомой категории → медиана / глобальное среднее.

## 3. Варианты датасетов
Лежат в `data/processed/` (parquet, в Git не хранятся — ссылка на Яндекс.Диск ниже).
Строки во всех вариантах выровнены, поэтому колонки можно свободно комбинировать.

| Вариант | Кодирование Region / Channel | Признаков | F1 на test_locked | Порог |
|---|---|---|---|---|
| v5_benchmark | нативный CatBoost (cat_features) | 10 | **0.4866** | 0.702 |
| v1_te | Target Encoding | 18 | 0.4856 | 0.698 |
| v4_woe | WOE | 16 | 0.4855 | 0.696 |
| v3_freq | Frequency | 16 | 0.4826 | 0.698 |
| v2_ohe | OHE (rare → Other, drop_first) | 45 | 0.4823 | 0.693 |

Все варианты, кроме v5, содержат агрегации и взаимодействия.
`test_locked.parquet` — финальный hold-out, использован один раз для таблицы выше.

**Данные:** [Яндекс.Диск — вставить ссылку]
Распаковать в `data/processed/`. Инструкция по исходному CSV — `data/DATA_SETUP.md`.

## 4. Как использовать
```python
from src.preprocessing import load_variant, combine_variants, scale_features

train, test, features, cat = load_variant("v5_benchmark")

# сборка своего набора: весь v5 + две колонки из v1
tr, te = combine_variants([("v5_benchmark", None),
                           ("v1_te", ["Region_Code_target_enc",
                                      "Policy_Sales_Channel_target_enc"])])
