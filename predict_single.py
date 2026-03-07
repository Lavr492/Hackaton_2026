import argparse, json
import numpy as np, pandas as pd, joblib
from catboost import CatBoostClassifier
from scipy.signal import savgol_filter

def savgol_row(arr, window=11, poly=3):
    try:
        return savgol_filter(arr, window_length=window, polyorder=poly)
    except Exception:
        return arr

def add_window_features(df, windows):
    X = df.copy()
    for a,b in windows:
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
# smoothing per row
df[feat_cols] = df[feat_cols].apply(lambda r: savgol_row(r.values, window=11, poly=3), axis=1, result_type='expand')
# add same windows as training (adjust if needed)
windows = [(450,500),(315,330)]
df = add_window_features(df, windows)

model = CatBoostClassifier()
model.load_model(args.model)
probs = model.predict_proba(df)
if args.thresholds:
    thresh = np.load(args.thresholds)
else:
    thresh = np.ones(probs.shape[1]) * 0.5

pred_idx = np.argmax(probs / thresh, axis=1)
if args.label_encoder:
    le = joblib.load(args.label_encoder)
    preds = le.classes_[pred_idx].tolist()
else:
    preds = model.classes_[pred_idx].tolist()

out = []
for i in range(len(preds)):
    out.append({'row': i, 'pred': preds[i], 'probs': probs[i].tolist()})
with open(args.out, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("Saved predictions to", args.out)
