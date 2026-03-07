from collections import Counter
from catboost import CatBoostClassifier, Pool

counts = Counter(y_train)
class_weights = [sum(counts.values())/counts[i] for i in sorted(counts)]

model = CatBoostClassifier(
    iterations=800,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=7,
    bootstrap_type='Bernoulli',   # чтобы subsample работал
    subsample=0.8,
    colsample_bylevel=0.8,
    class_weights=class_weights,
    random_seed=42,
    verbose=100,
    train_dir='catboost_tuned_logs',
    use_best_model=True
)

model.fit(Pool(X_train, y_train), eval_set=Pool(X_val, y_val), early_stopping_rounds=50)
model.save_model("catboost_tuned.cbm")
