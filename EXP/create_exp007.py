import os
import nbformat

ROOT = r"D:/program/Kaggle/Kaggle-Predicting-Stellar-Class"
SRC = os.path.join(ROOT, "EXP", "EXP005", "stellar_classification.ipynb")
OUT_DIR = os.path.join(ROOT, "EXP", "EXP007")
OUT_NB = os.path.join(OUT_DIR, "stellar_classification.ipynb")

os.makedirs(OUT_DIR, exist_ok=True)

nb = nbformat.read(SRC, as_version=4)

cell1 = nb.cells[1]["source"]
cell1 = cell1.replace(
    "from sklearn.ensemble import ExtraTreesClassifier",
    "from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier",
)
cell1 = cell1.replace(
    "from sklearn.linear_model import LogisticRegression",
    "from sklearn.linear_model import LogisticRegression\nfrom sklearn.neighbors import KNeighborsClassifier",
)
cell1 = cell1.replace(
    'SUBMISSION_FILENAME = "submission-005.csv"',
    'SUBMISSION_FILENAME = "submission-007.csv"',
)
nb.cells[1]["source"] = cell1

nb.cells[7]["source"] = "## 3. 学習(5-fold CV + 20モデル)"

nb.cells[8]["source"] = """from catboost import Pool

N_CLASSES = len(target_encoder.classes_)

MODEL_SPECS = [
    {"name": "lgbm_1", "family": "lgbm", "params": {"n_estimators": 1800, "learning_rate": 0.05, "num_leaves": 127, "colsample_bytree": 0.8, "subsample": 0.8, "min_child_samples": 30}},
    {"name": "lgbm_2", "family": "lgbm", "params": {"n_estimators": 2200, "learning_rate": 0.04, "num_leaves": 255, "colsample_bytree": 0.9, "subsample": 0.9, "min_child_samples": 40}},
    {"name": "lgbm_3", "family": "lgbm", "params": {"n_estimators": 1600, "learning_rate": 0.06, "num_leaves": 63, "colsample_bytree": 0.7, "subsample": 0.8, "min_child_samples": 60}},
    {"name": "lgbm_4", "family": "lgbm", "params": {"n_estimators": 2600, "learning_rate": 0.03, "num_leaves": 191, "colsample_bytree": 0.85, "subsample": 0.9, "min_child_samples": 25}},
    {"name": "lgbm_goss", "family": "lgbm", "params": {"n_estimators": 2000, "learning_rate": 0.05, "num_leaves": 127, "boosting_type": "goss", "top_rate": 0.2, "other_rate": 0.1, "min_child_samples": 30}},
    {"name": "xgb_1", "family": "xgb", "params": {"n_estimators": 1800, "learning_rate": 0.05, "max_depth": 8, "colsample_bytree": 0.8, "subsample": 0.8, "min_child_weight": 1}},
    {"name": "xgb_2", "family": "xgb", "params": {"n_estimators": 2200, "learning_rate": 0.04, "max_depth": 10, "colsample_bytree": 0.9, "subsample": 0.9, "min_child_weight": 2}},
    {"name": "xgb_3", "family": "xgb", "params": {"n_estimators": 1500, "learning_rate": 0.06, "max_depth": 6, "colsample_bytree": 0.7, "subsample": 0.8, "gamma": 0.2}},
    {"name": "xgb_4", "family": "xgb", "params": {"n_estimators": 2600, "learning_rate": 0.03, "max_depth": 9, "colsample_bytree": 0.85, "subsample": 0.9, "reg_lambda": 2.0}},
    {"name": "cat_1", "family": "cat", "params": {"iterations": 2000, "learning_rate": 0.08, "depth": 8, "l2_leaf_reg": 3.0}},
    {"name": "cat_2", "family": "cat", "params": {"iterations": 2500, "learning_rate": 0.05, "depth": 10, "l2_leaf_reg": 5.0}},
    {"name": "cat_3", "family": "cat", "params": {"iterations": 1600, "learning_rate": 0.10, "depth": 6, "random_strength": 2.0}},
    {"name": "et_1", "family": "et", "params": {"n_estimators": 700, "max_depth": None, "min_samples_leaf": 2, "max_features": "sqrt"}},
    {"name": "rf_1", "family": "rf", "params": {"n_estimators": 600, "max_depth": None, "min_samples_leaf": 2, "max_features": "sqrt", "class_weight": "balanced_subsample"}},
    {"name": "hgb_1", "family": "hgb", "params": {"learning_rate": 0.05, "max_iter": 600, "max_depth": 8, "min_samples_leaf": 20}},
    {"name": "hgb_2", "family": "hgb", "params": {"learning_rate": 0.03, "max_iter": 900, "max_depth": 10, "min_samples_leaf": 30}},
    {"name": "logreg_1", "family": "logreg", "params": {"C": 2.0, "max_iter": 4000}},
    {"name": "logreg_2", "family": "logreg", "params": {"C": 0.8, "max_iter": 5000}},
    {"name": "knn_1", "family": "knn", "params": {"n_neighbors": 35, "weights": "distance"}},
    {"name": "knn_2", "family": "knn", "params": {"n_neighbors": 55, "weights": "distance"}},
]

MODEL_NAMES = [m["name"] for m in MODEL_SPECS]
MODEL_MAP = {m["name"]: m for m in MODEL_SPECS}

oof = {m: np.zeros((len(X), N_CLASSES)) for m in MODEL_NAMES}
pred_test = {m: np.zeros((len(X_test), N_CLASSES)) for m in MODEL_NAMES}
oof_fold = {m: [] for m in MODEL_NAMES}
pred_test_fold = {m: [] for m in MODEL_NAMES}
va_indices = []
y_valid_folds = []

cat_feature_idx = [FEATURES.index(c) for c in CAT_COLS]

for seed in SEED_LIST:
    print(f"=== seed {seed} ===")
    oof_seed = {m: np.zeros((len(X), N_CLASSES)) for m in MODEL_NAMES}
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        w_tr, w_va = sample_weight[tr_idx], sample_weight[va_idx]

        for model_name in MODEL_NAMES:
            spec = MODEL_MAP[model_name]
            family = spec["family"]
            p = dict(spec["params"])

            if family == "lgbm":
                model = lgb.LGBMClassifier(
                    objective="multiclass",
                    random_state=seed,
                    n_jobs=-1,
                    verbosity=-1,
                    **p,
                )
                model.fit(
                    X_tr, y_tr,
                    sample_weight=w_tr,
                    eval_set=[(X_va, y_va)],
                    eval_sample_weight=[w_va],
                    categorical_feature=CAT_COLS,
                    callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
                )
            elif family == "xgb":
                model = xgb.XGBClassifier(
                    objective="multi:softprob",
                    tree_method="hist",
                    device="cuda",
                    early_stopping_rounds=80,
                    random_state=seed,
                    n_jobs=-1,
                    verbosity=0,
                    **p,
                )
                model.fit(
                    X_tr, y_tr,
                    sample_weight=w_tr,
                    eval_set=[(X_va, y_va)],
                    sample_weight_eval_set=[w_va],
                    verbose=False,
                )
            elif family == "cat":
                model = CatBoostClassifier(
                    loss_function="MultiClass",
                    random_seed=seed,
                    early_stopping_rounds=80,
                    task_type="GPU",
                    devices="0",
                    verbose=0,
                    **p,
                )
                model.fit(
                    Pool(X_tr, y_tr, weight=w_tr, cat_features=cat_feature_idx),
                    eval_set=Pool(X_va, y_va, weight=w_va, cat_features=cat_feature_idx),
                )
            elif family == "et":
                model = ExtraTreesClassifier(random_state=seed, n_jobs=-1, **p)
                model.fit(X_tr, y_tr, sample_weight=w_tr)
            elif family == "rf":
                model = RandomForestClassifier(random_state=seed, n_jobs=-1, **p)
                model.fit(X_tr, y_tr, sample_weight=w_tr)
            elif family == "hgb":
                model = HistGradientBoostingClassifier(random_state=seed, **p)
                model.fit(X_tr, y_tr, sample_weight=w_tr)
            elif family == "logreg":
                model = LogisticRegression(
                    multi_class="multinomial",
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=seed,
                    n_jobs=-1,
                    **p,
                )
                model.fit(X_tr, y_tr, sample_weight=w_tr)
            elif family == "knn":
                model = KNeighborsClassifier(**p)
                model.fit(X_tr, y_tr)
            else:
                raise ValueError(f"unknown family: {family}")

            va_pred = model.predict_proba(X_va)
            te_pred = model.predict_proba(X_test)

            oof_seed[model_name][va_idx] = va_pred
            pred_test[model_name] += te_pred / (N_FOLDS * len(SEED_LIST))
            oof_fold[model_name].append(va_pred)
            pred_test_fold[model_name].append(te_pred)

        va_indices.append(va_idx)
        y_valid_folds.append(y_va)

        fold_scores = {m: balanced_accuracy_score(y_va, oof_seed[m][va_idx].argmax(axis=1)) for m in MODEL_NAMES}
        top5 = sorted(fold_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"seed {seed} fold {fold} top5:", ", ".join(f"{m}={s:.5f}" for m, s in top5))

    for m in MODEL_NAMES:
        oof[m] += oof_seed[m] / len(SEED_LIST)

for m in MODEL_NAMES:
    print(f"OOF balanced accuracy [{m}] seed-avg: {balanced_accuracy_score(y, oof[m].argmax(axis=1)):.5f}")
"""

nb.cells[10]["source"] = """from itertools import product

# Fast ensemble for many models:
# 1) score each model's OOF
model_scores = {m: balanced_accuracy_score(y, oof[m].argmax(axis=1)) for m in MODEL_NAMES}
score_sum = sum(max(v, 1e-6) for v in model_scores.values())

# 2) normalize to weights (acts as a strong baseline with 20 models)
best_weights = tuple(max(model_scores[m], 1e-6) / score_sum for m in MODEL_NAMES)
best_score = balanced_accuracy_score(y, sum(w * oof[m] for w, m in zip(best_weights, MODEL_NAMES)).argmax(axis=1))

print("n_models:", len(MODEL_NAMES))
print("top10 model OOF scores:", sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:10])
print(f"ensemble OOF balanced accuracy (base) = {best_score:.5f}")

def find_best_bias(log_proba, y_true, bias_grid):
    best_bias = np.zeros(log_proba.shape[1])
    best_score_local = -1.0
    for bias in product(bias_grid, repeat=log_proba.shape[1]):
        bias = np.array(bias)
        pred_tmp = (log_proba + bias).argmax(axis=1)
        s = balanced_accuracy_score(y_true, pred_tmp)
        if s > best_score_local:
            best_score_local = s
            best_bias = bias.copy()
    return best_bias, best_score_local

def apply_log_bias_to_proba(proba, bias):
    adj = np.exp(np.log(np.clip(proba, 1e-15, 1.0)) + bias)
    return adj / adj.sum(axis=1, keepdims=True)

oof_blend_base = sum(w * oof[m] for w, m in zip(best_weights, MODEL_NAMES))
bias_grid = np.arange(-0.06, 0.061, 0.01)
fold_biases = []
oof_blend_bias = np.zeros((len(X), N_CLASSES))
test_blend_bias = np.zeros((len(X_test), N_CLASSES))
n_valid_folds = len(va_indices)

for fold in range(n_valid_folds):
    va_idx = va_indices[fold]
    y_va = y_valid_folds[fold]
    va_blend = sum(w * oof_fold[m][fold] for w, m in zip(best_weights, MODEL_NAMES))
    test_blend_fold = sum(w * pred_test_fold[m][fold] for w, m in zip(best_weights, MODEL_NAMES))
    best_bias, best_score_fold = find_best_bias(np.log(np.clip(va_blend, 1e-15, 1.0)), y_va, bias_grid)
    fold_biases.append(best_bias)
    oof_blend_bias[va_idx] = apply_log_bias_to_proba(va_blend, best_bias)
    test_blend_bias += apply_log_bias_to_proba(test_blend_fold, best_bias) / n_valid_folds
    if fold < 5:
        print(f"fold {fold} best bias = {best_bias.tolist()}, score = {best_score_fold:.5f}")

oof_pred = oof_blend_bias.argmax(axis=1)
score_bias = balanced_accuracy_score(y, oof_pred)
print(f"ensemble OOF balanced accuracy (bias-corrected) = {score_bias:.5f}")
print("mean fold bias:", np.mean(np.stack(fold_biases), axis=0).tolist())

test_blend = test_blend_bias
cm = confusion_matrix(y, oof_pred, normalize="true")
disp = ConfusionMatrixDisplay(cm, display_labels=target_encoder.classes_)
disp.plot(cmap="Blues", values_format=".3f")
plt.title("OOF confusion matrix (row-normalized, bias-corrected)")
plt.show()
"""

nbformat.write(nb, OUT_NB)

with open(os.path.join(OUT_DIR, "README.md"), "w", encoding="utf-8") as f:
    f.write(
        "# EXP007\\n\\n"
        "## 変更点\\n"
        "- EXP005をベースに学習モデルを20本へ拡張\\n"
        "- モデル群: LGBM(5) / XGB(4) / Cat(3) / ET(1) / RF(1) / HGB(2) / Logistic(2) / KNN(2)\\n"
        "- 20モデル向けに、重み探索は全探索ではなくOOFスコア正規化で高速化\\n"
        "- 出力ファイル名: submission-007.csv\\n"
    )

print("created", OUT_NB)
