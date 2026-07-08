import numpy as np
import json
from dataset import get_dataset
from features import extract_features, normalize_matrix, NUM_FEATURES

np.random.seed(42)

# ── Load & split data ──────────────────────────────────────
data = get_dataset()
np.random.shuffle(data)

texts   = [d[0] for d in data]
levels  = np.array([d[1] for d in data], dtype=int)
scores  = np.array([d[2] for d in data], dtype=float)
stressed= np.array([1 if d[3] else 0 for d in data], dtype=int)

X_raw = np.array([extract_features(t) for t in texts], dtype=float)

n = len(data)
n_test = max(int(n * 0.2), 8)
test_idx = list(range(n - n_test, n))
train_idx = list(range(0, n - n_test))

X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
y_train_score, y_test_score = scores[train_idx], scores[test_idx]
y_train_lvl, y_test_lvl = levels[train_idx], levels[test_idx]
y_train_svm, y_test_svm = stressed[train_idx], stressed[test_idx]

X_train, mean, std = normalize_matrix(X_train_raw)
X_test, _, _ = normalize_matrix(X_test_raw, mean, std)

print(f"Train size: {len(train_idx)}  Test size: {len(test_idx)}")

# ══════════════════════════════════════════════════════════
# LINEAR REGRESSION — batch gradient descent, from scratch
# ══════════════════════════════════════════════════════════
def train_linear_regression(X, y, lr=0.05, epochs=3000, l2=0.01):
    n_samples, n_feat = X.shape
    w = np.zeros(n_feat)
    b = 0.0
    for _ in range(epochs):
        pred = X.dot(w) + b
        err = pred - y
        grad_w = (X.T.dot(err) / n_samples) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b

lr_w, lr_b = train_linear_regression(X_train, y_train_score)

def lr_predict(X, w, b):
    return np.clip(X.dot(w) + b, 0, 10)

train_pred = lr_predict(X_train, lr_w, lr_b)
test_pred = lr_predict(X_test, lr_w, lr_b)
mae_test = np.mean(np.abs(test_pred - y_test_score))
rmse_test = np.sqrt(np.mean((test_pred - y_test_score) ** 2))
ss_res = np.sum((y_test_score - test_pred) ** 2)
ss_tot = np.sum((y_test_score - y_test_score.mean()) ** 2)
r2_test = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

print(f"\n[Linear Regression]  MAE={mae_test:.3f}  RMSE={rmse_test:.3f}  R2={r2_test:.3f}")

# ══════════════════════════════════════════════════════════
# LINEAR SVM — hinge loss + subgradient descent, from scratch
# ══════════════════════════════════════════════════════════
def train_svm(X, y01, lr=0.01, epochs=3000, C=1.0):
    y = np.where(y01 == 1, 1.0, -1.0)
    n_samples, n_feat = X.shape
    w = np.zeros(n_feat)
    b = 0.0
    for _ in range(epochs):
        margins = y * (X.dot(w) + b)
        mask = (margins < 1).astype(float)
        grad_w = w - C * (X.T.dot(mask * y)) / n_samples
        grad_b = -C * np.sum(mask * y) / n_samples
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b

svm_w, svm_b = train_svm(X_train, y_train_svm)

def svm_predict(X, w, b):
    return (X.dot(w) + b > 0).astype(int)

svm_pred_test = svm_predict(X_test, svm_w, svm_b)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def f1_binary(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0, prec, rec

svm_acc = accuracy(y_test_svm, svm_pred_test)
svm_f1, svm_prec, svm_rec = f1_binary(y_test_svm, svm_pred_test)
print(f"[SVM]                Accuracy={svm_acc:.3f}  F1={svm_f1:.3f}  Precision={svm_prec:.3f}  Recall={svm_rec:.3f}")

# ══════════════════════════════════════════════════════════
# DECISION TREE — greedy Gini-impurity splits, from scratch
# ══════════════════════════════════════════════════════════
def gini(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1 - np.sum(p ** 2)

def best_split(X, y):
    n_samples, n_feat = X.shape
    best_gain, best_feat, best_thresh = -1, None, None
    parent_gini = gini(y)
    for feat in range(n_feat):
        thresholds = np.unique(X[:, feat])
        for t in thresholds:
            left_mask = X[:, feat] <= t
            right_mask = ~left_mask
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue
            gl, gr = gini(y[left_mask]), gini(y[right_mask])
            weighted = (left_mask.sum() * gl + right_mask.sum() * gr) / n_samples
            gain = parent_gini - weighted
            if gain > best_gain:
                best_gain, best_feat, best_thresh = gain, feat, t
    return best_feat, best_thresh, best_gain

def build_tree(X, y, depth=0, max_depth=4, min_samples=3):
    if depth >= max_depth or len(y) < min_samples or gini(y) == 0:
        vals, counts = np.unique(y, return_counts=True)
        return {"leaf": True, "class": int(vals[np.argmax(counts)])}
    feat, thresh, gain = best_split(X, y)
    if feat is None or gain <= 1e-9:
        vals, counts = np.unique(y, return_counts=True)
        return {"leaf": True, "class": int(vals[np.argmax(counts)])}
    left_mask = X[:, feat] <= thresh
    right_mask = ~left_mask
    return {
        "leaf": False, "feature": feat, "threshold": float(thresh),
        "left": build_tree(X[left_mask], y[left_mask], depth + 1, max_depth, min_samples),
        "right": build_tree(X[right_mask], y[right_mask], depth + 1, max_depth, min_samples),
    }

def tree_predict_one(node, x):
    while not node["leaf"]:
        node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
    return node["class"]

def tree_predict(tree, X):
    return np.array([tree_predict_one(tree, x) for x in X])

dt_tree = build_tree(X_train, y_train_lvl, max_depth=6, min_samples=2)
dt_pred_train = tree_predict(dt_tree, X_train)
dt_pred_test = tree_predict(dt_tree, X_test)
dt_acc_train = accuracy(y_train_lvl, dt_pred_train)
dt_acc = accuracy(y_test_lvl, dt_pred_test)
print(f"[Decision Tree] train_acc={dt_acc_train:.3f}")
print("train level distribution:", np.bincount(y_train_lvl))
print("test level distribution:", np.bincount(y_test_lvl))
print("test pred distribution:", np.bincount(dt_pred_test, minlength=4))

def f1_macro(y_true, y_pred, classes):
    f1s = []
    for c in classes:
        yt = (y_true == c).astype(int)
        yp = (y_pred == c).astype(int)
        f1, _, _ = f1_binary(yt, yp)
        f1s.append(f1)
    return np.mean(f1s)

dt_f1 = f1_macro(y_test_lvl, dt_pred_test, [0, 1, 2, 3])
dt_adj_acc = np.mean(np.abs(y_test_lvl - dt_pred_test) <= 1)  # ordinal off-by-one tolerance
print(f"[Decision Tree]      Accuracy={dt_acc:.3f}  Macro-F1={dt_f1:.3f}  Adjacent(±1)Accuracy={dt_adj_acc:.3f}")

# ══════════════════════════════════════════════════════════
# ENSEMBLE — evaluate combined vote against true levels
# ══════════════════════════════════════════════════════════
def score_to_level(s):
    if s >= 7: return 3
    if s >= 5: return 2
    if s >= 3: return 1
    return 0

ens_scores = 0.5 * test_pred + (10/3) * 0.3 * dt_pred_test + 10 * 0.2 * svm_pred_test
ens_scores = np.clip(ens_scores, 0, 10)
ens_levels = np.array([score_to_level(s) for s in ens_scores])
ens_acc = accuracy(y_test_lvl, ens_levels)
ens_f1 = f1_macro(y_test_lvl, ens_levels, [0, 1, 2, 3])
print(f"[Ensemble]           Accuracy={ens_acc:.3f}  Macro-F1={ens_f1:.3f}")

# ── Save everything the app needs ──────────────────────────
out = {
    "feature_mean": mean.tolist(),
    "feature_std": std.tolist(),
    "lr_w": lr_w.tolist(),
    "lr_b": float(lr_b),
    "svm_w": svm_w.tolist(),
    "svm_b": float(svm_b),
    "dt_tree": dt_tree,
    "metrics": {
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "lr_mae": round(float(mae_test), 3),
        "lr_rmse": round(float(rmse_test), 3),
        "lr_r2": round(float(r2_test), 3),
        "svm_accuracy": round(float(svm_acc), 3),
        "svm_f1": round(float(svm_f1), 3),
        "svm_precision": round(float(svm_prec), 3),
        "svm_recall": round(float(svm_rec), 3),
        "dt_accuracy": round(float(dt_acc), 3),
        "dt_f1": round(float(dt_f1), 3),
        "dt_adjacent_accuracy": round(float(dt_adj_acc), 3),
        "ensemble_accuracy": round(float(ens_acc), 3),
        "ensemble_f1": round(float(ens_f1), 3),
    }
}
with open("trained_params.json", "w") as f:
    json.dump(out, f, indent=2)

print("\nSaved trained_params.json")
print(json.dumps(out["metrics"], indent=2))
