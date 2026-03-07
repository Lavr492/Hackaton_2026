# Hackaton_2026
Рамановские спектры
### О проекте
Классификация спектров по трём классам: control, endo, exo. Репозиторий содержит воспроизводимый пайплайн: загрузка данных, предобработка (сглаживание, агрегаты по окнам), PCA‑анализ, обучение CatBoost, оптимизация порогов, оценка (confusion matrix, feature importance) и утилиты для предсказания на одном спектре.

---

### Быстрое воспроизведение для проверяющего
1. Установить зависимости:
   ```bash
   pip install -r requirements.txt
### Положить данные в data/:
основной набор: data/train.csv (строки — образцы; колонки v0,v1,...,vN; колонка меток label).
одиночный спектр для проверки: data/single_spectrum.csv (см. пример ниже).
Быстрый прогон обучения (малые итерации):
bash

python src/models/train_catboost.py --config experiments/catboost_quick.yaml
или запустить единый скрипт для проверяющего:
bash

python run_model_for_reviewer.py --data data/train.csv --target label --out_dir results --quick
Предсказание на одном спектре:
bash

python src/predict_single.py --model results/catboost_model.cbm --label-encoder results/label_encoder.joblib --thresholds results/best_thresholds.npy --input data/single_spectrum.csv --out results/prediction.json
Просмотреть артефакты:
results/confusion_matrix_test.png — матрица ошибок;
results/feature_importance.csv — важности признаков;
results/classification_report_test.txt — precision/recall/f1;
results/best_thresholds.npy — подобранные пороги.   
