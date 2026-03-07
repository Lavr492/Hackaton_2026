#!/usr/bin/env python3
import yaml, argparse
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from collections import Counter
from pathlib import Path
import joblib

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def add_window_features(df, windows):
    X = df.copy()
    for (a,b) in windows:
        cols = [f'v{i}' for i in range(a, b+1) if f'v{i}' in X.columns]
        if not cols: continue
        X[f'int_{a}_{b}'] = X[cols].sum(axis=1)
        X[f'max_{a}_{b}'] = X[cols].max(axis=1)
        X[f'mean_{a}_{b}'] = X[cols].mean(axis=1)
    return X

def compute_class_weights(y):
    counts = Counter(y)
    total = sum(counts.values())
    classes = sorted(counts.keys())
    return [total/counts[c] for c in classes]

def main(config_path):
    cfg = load_config(config_path)
    df = pd.read_csv(cfg['data']['path'])
    target = cfg['data']['target']
    X = df[[c for c in df.columns if c.startswith('v')]]
    y = df[target]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=cfg['split']['test_size'], stratify=y, random_state=42)
    val_rel = cfg['split']['val_size'] / (1 - cfg['split']['test_size'])
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_rel, stratify=y_temp, random_state=42)
    X_train = add_window_features(X_train, cfg['features']['windows'])
    X_val = add_window_features(X_val, cfg['features']['windows'])
    class_weights = compute_class_weights(y_train)
    model = CatBoostClassifier(
        iterations=cfg['catboost']['iterations'],
        learning_rate=cfg['catboost']['learning_rate'],
        depth=cfg['catboost']['depth'],
        l2_leaf_reg=cfg['catboost']['l2_leaf_reg'],
        subsample=cfg['catboost']['subsample'],
        colsample_bylevel=cfg['catboost']['colsample_bylevel'],
        class_weights=class_weights,
        random_seed=42,
        verbose=100,
        use_best_model=True
    )
    model.fit(Pool(X_train, y_train), eval_set=Pool(X_val, y_val), early_stopping_rounds=cfg['catboost']['early_stopping_rounds'])
    out_path = Path(cfg['output']['model_path'])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_path))
    joblib.dump(list(X_train.columns), out_path.parent / "feature_names.joblib")
    print("Saved model to", out_path)

if name == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/catboost_quick.yaml")
    args = parser.parse_args()
    main(args.config)
