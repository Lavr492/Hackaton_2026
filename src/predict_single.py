#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd, numpy as np, joblib
from catboost import CatBoostClassifier
from scipy.signal import savgol_filter

def add_window_features(df, windows):
    X = df.copy()
    for (a,b) in windows:
        cols = [f'v{i}' for i in range(a, b+1) if f'v{i}' in X.columns]
        if not cols: continue
        X[f'int_{a}_{b}'] = X[cols].sum(axis=1)
        X[f'max_{a}_{b}'] = X[cols].max(axis=1)
        X[f'mean_{a}_{b}'] = X[cols].mean(axis=1)
    return X

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--label-encoder', required=False)
parser.add_argument('--thresholds', required=False)
parser.add_argument('--input', required=True)
parser.add_argument('--out', default='results/prediction.json')
args = parser.parse_args()

df = pd.read_csv(args.input)
feat_cols = [c for c in df.columns if c.startswith('v')]
try:
    df[feat_cols] = df[feat_cols].apply(lambda r: savgol_filter(r.values, window_length=11, polyorder=3), axis=1, result_type='expand')
except Exception:
    pass
df = add_window_features(df, [(450,500),(315,330)])
model = CatBoostClassifier()
model.load_model(args.model)
probs = model.predict_proba(df)
if args.thresholds:
    thresh = np.load(args.thresholds)
else:
    thresh = np.ones(probs.shape[1])*0.5
pred_idx = np.argmax(probs / thresh, axis=1)
if args.label_encoder:
    le = joblib.load(args.label_encoder)
    preds = le.classes_[pred_idx].tolist()
else:
    try:
        preds = model.classes_[pred_idx].tolist()
    except Exception:
        preds = [str(i) for i in pred_idx]
out = [{"row":int(i),"pred":preds[i],"probs":probs[i].tolist()} for i in range(len(preds))]
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
print("Saved predictions to", args.out)
