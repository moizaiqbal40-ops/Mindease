import csv
import numpy as np
from features import extract_features, normalize_matrix

np.random.seed(42)

def load_dreaddit(path):
    with open(path, encoding="latin-1") as f:
        rows = list(csv.DictReader(f))
    texts = [r["text"] for r in rows]
    labels = np.array([int(r["label"]) for r in rows], dtype=int)
    return texts, labels

train_texts, y_train = load_dreaddit("dreaddit_train.csv")
test_texts, y_test = load_dreaddit("dreaddit_test.csv")

print(f"Dreaddit train: {len(train_texts)}  test: {len(test_texts)}")
print(f"Train label balance: {y_train.mean():.3f} stressed")
print(f"Test  label balance: {y_test.mean():.3f} stressed")

X_train_raw = np.array([extract_features(t) for t in train_texts], dtype=float)
X_test_raw  = np.array([extract_features(t) for t in test_texts], dtype=float)

X_train, mean, std = normalize_matrix(X_train_raw)
X_test, _, _ = normalize_matrix(X_test_raw, mean, std)

def train_svm(X, y01, lr=0.01, epochs=2000, C=1.0):
    y = np.where(y01 == 1, 1.0, -1.0)
    n_samples, n_feat = X.shape
    w = np.zeros(n_feat)
    b = 0.0
    for epoch in range(epochs):
        margins = y * (X.dot(w) + b)
        mask = (margins < 1).astype(float)
        grad_w = w - C * (X.T.dot(mask * y)) / n_samples
        grad_b = -C * np.sum(mask * y) / n_samples
        w -= lr * grad_w
        b -= lr * grad_b
        if epoch % 400 == 0:
            loss = 0.5*np.dot(w,w) + C*np.mean(np.maximum(0, 1-margins))
            print(f"  epoch {epoch}: hinge loss {loss:.4f}")
    return w, b

print("\nTraining linear SVM (hinge loss, subgradient descent) on real Dreaddit data...")
svm_w, svm_b = train_svm(X_train, y_train)

def svm_predict(X, w, b):
    return (X.dot(w) + b > 0).astype(int)

pred_test = svm_predict(X_test, svm_w, svm_b)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def f1_binary(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0), prec, rec

acc = accuracy(y_test, pred_test)
f1, prec, rec = f1_binary(y_test, pred_test)
print(f"\n[SVM on Dreaddit test set — {len(test_texts)} real Reddit posts, held out per the original paper's split]")
print(f"Accuracy={acc:.3f}  F1={f1:.3f}  Precision={prec:.3f}  Recall={rec:.3f}")

import json
out = {
    "feature_mean": mean.tolist(), "feature_std": std.tolist(),
    "svm_w": svm_w.tolist(), "svm_b": float(svm_b),
    "metrics": {"n_train": len(train_texts), "n_test": len(test_texts),
                "accuracy": round(float(acc),3), "f1": round(float(f1),3),
                "precision": round(float(prec),3), "recall": round(float(rec),3)}
}
with open("dreaddit_svm_params.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved dreaddit_svm_params.json")
