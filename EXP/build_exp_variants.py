import os
from copy import deepcopy

import nbformat


ROOT = "D:/program/Kaggle/Kaggle-Predicting-Stellar-Class"
BASE_NB = os.path.join(ROOT, "EXP", "EVP001", "stellar_classification.ipynb")


CELL1_COMMON = """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

SEED = 42
N_FOLDS = 5
SEED_LIST = [42, 52, 62]

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
sample_sub = pd.read_csv("data/sample_submission.csv")

print(train.shape, test.shape)
train.head()
"""


CELL8_SEED_AVG = """from catboost import Pool

N_CLASSES = len(target_encoder.classes_)
MODEL_NAMES = ["lgbm", "xgb", "cat", "et", "lgbm_alt", "xgb_alt"]

oof = {m: np.zeros((len(X), N_CLASSES)) for m in MODEL_NAMES}
pred_test = {m: np.zeros((len(X_test), N_CLASSES)) for m in MODEL_NAMES}

# seed x fold predictions for fold-wise post processing
oof_fold = {m: [] for m in MODEL_NAMES}
pred_test_fold = {m: [] for m in MODEL_NAMES}
va_indices = []
y_valid_folds = []

for seed in SEED_LIST:
    print(f"=== seed {seed} ===")
    oof_seed = {m: np.zeros((len(X), N_CLASSES)) for m in MODEL_NAMES}
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        w_tr, w_va = sample_weight[tr_idx], sample_weight[va_idx]

        # ---- LightGBM ----
        lgbm_model = lgb.LGBMClassifier(
            objective="multiclass",
            n_estimators=2000,
            learning_rate=0.05,
            num_leaves=127,
            colsample_bytree=0.8,
            subsample=0.8,
            subsample_freq=1,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        lgbm_model.fit(
            X_tr,
            y_tr,
            sample_weight=w_tr,
            eval_set=[(X_va, y_va)],
            eval_sample_weight=[w_va],
            categorical_feature=CAT_COLS,
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        lgbm_va = lgbm_model.predict_proba(X_va)
        lgbm_test = lgbm_model.predict_proba(X_test)
        oof_seed["lgbm"][va_idx] = lgbm_va
        pred_test["lgbm"] += lgbm_test / (N_FOLDS * len(SEED_LIST))
        oof_fold["lgbm"].append(lgbm_va)
        pred_test_fold["lgbm"].append(lgbm_test)

        # ---- XGBoost ----
        xgb_model = xgb.XGBClassifier(
            objective="multi:softprob",
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=8,
            colsample_bytree=0.8,
            subsample=0.8,
            tree_method="hist",
            device="cuda",
            early_stopping_rounds=100,
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
        xgb_model.fit(
            X_tr,
            y_tr,
            sample_weight=w_tr,
            eval_set=[(X_va, y_va)],
            sample_weight_eval_set=[w_va],
            verbose=False,
        )
        xgb_va = xgb_model.predict_proba(X_va)
        xgb_test = xgb_model.predict_proba(X_test)
        oof_seed["xgb"][va_idx] = xgb_va
        pred_test["xgb"] += xgb_test / (N_FOLDS * len(SEED_LIST))
        oof_fold["xgb"].append(xgb_va)
        pred_test_fold["xgb"].append(xgb_test)

        # ---- CatBoost ----
        cat_feature_idx = [FEATURES.index(c) for c in CAT_COLS]
        cat_model = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=2000,
            learning_rate=0.08,
            depth=8,
            random_seed=seed,
            early_stopping_rounds=100,
            task_type="GPU",
            devices="0",
            verbose=0,
        )
        cat_model.fit(
            Pool(X_tr, y_tr, weight=w_tr, cat_features=cat_feature_idx),
            eval_set=Pool(X_va, y_va, weight=w_va, cat_features=cat_feature_idx),
        )
        cat_va = cat_model.predict_proba(X_va)
        cat_test = cat_model.predict_proba(X_test)
        oof_seed["cat"][va_idx] = cat_va
        pred_test["cat"] += cat_test / (N_FOLDS * len(SEED_LIST))
        oof_fold["cat"].append(cat_va)
        pred_test_fold["cat"].append(cat_test)

        # ---- ExtraTrees ----
        et_model = ExtraTreesClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        )
        et_model.fit(X_tr, y_tr, sample_weight=w_tr)
        et_va = et_model.predict_proba(X_va)
        et_test = et_model.predict_proba(X_test)
        oof_seed["et"][va_idx] = et_va
        pred_test["et"] += et_test / (N_FOLDS * len(SEED_LIST))
        oof_fold["et"].append(et_va)
        pred_test_fold["et"].append(et_test)

        # ---- LightGBM (alt) ----
        lgbm_alt_model = lgb.LGBMClassifier(
            objective="multiclass",
            n_estimators=2200,
            learning_rate=0.04,
            num_leaves=255,
            max_depth=-1,
            min_child_samples=40,
            colsample_bytree=0.9,
            subsample=0.9,
            subsample_freq=1,
            random_state=seed + 101,
            n_jobs=-1,
            verbosity=-1,
        )
        lgbm_alt_model.fit(
            X_tr,
            y_tr,
            sample_weight=w_tr,
            eval_set=[(X_va, y_va)],
            eval_sample_weight=[w_va],
            categorical_feature=CAT_COLS,
            callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)],
        )
        lgbm_alt_va = lgbm_alt_model.predict_proba(X_va)
        lgbm_alt_test = lgbm_alt_model.predict_proba(X_test)
        oof_seed["lgbm_alt"][va_idx] = lgbm_alt_va
        pred_test["lgbm_alt"] += lgbm_alt_test / (N_FOLDS * len(SEED_LIST))
        oof_fold["lgbm_alt"].append(lgbm_alt_va)
        pred_test_fold["lgbm_alt"].append(lgbm_alt_test)

        # ---- XGBoost (alt) ----
        xgb_alt_model = xgb.XGBClassifier(
            objective="multi:softprob",
            n_estimators=2200,
            learning_rate=0.04,
            max_depth=10,
            min_child_weight=2,
            gamma=0.0,
            colsample_bytree=0.9,
            subsample=0.9,
            tree_method="hist",
            device="cuda",
            early_stopping_rounds=120,
            random_state=seed + 202,
            n_jobs=-1,
            verbosity=0,
        )
        xgb_alt_model.fit(
            X_tr,
            y_tr,
            sample_weight=w_tr,
            eval_set=[(X_va, y_va)],
            sample_weight_eval_set=[w_va],
            verbose=False,
        )
        xgb_alt_va = xgb_alt_model.predict_proba(X_va)
        xgb_alt_test = xgb_alt_model.predict_proba(X_test)
        oof_seed["xgb_alt"][va_idx] = xgb_alt_va
        pred_test["xgb_alt"] += xgb_alt_test / (N_FOLDS * len(SEED_LIST))
        oof_fold["xgb_alt"].append(xgb_alt_va)
        pred_test_fold["xgb_alt"].append(xgb_alt_test)

        va_indices.append(va_idx)
        y_valid_folds.append(y_va)

        scores = {
            m: balanced_accuracy_score(y_va, oof_seed[m][va_idx].argmax(axis=1))
            for m in MODEL_NAMES
        }
        print(f"seed {seed} fold {fold}: " + ", ".join(f"{m}={s:.5f}" for m, s in scores.items()))

    for m in MODEL_NAMES:
        oof[m] += oof_seed[m] / len(SEED_LIST)
        print(f"seed {seed} OOF [{m}]: {balanced_accuracy_score(y, oof_seed[m].argmax(axis=1)):.5f}")

for m in MODEL_NAMES:
    print(f"OOF balanced accuracy [{m}] seed-avg: {balanced_accuracy_score(y, oof[m].argmax(axis=1)):.5f}")
"""


CELL10_SEED_AVG = """from itertools import product

# 1) Optimize model weights on OOF
def generate_weight_candidates(num_models, total_steps=10):
    if num_models == 1:
        yield (total_steps,)
        return
    for i in range(total_steps + 1):
        for tail in generate_weight_candidates(num_models - 1, total_steps - i):
            yield (i,) + tail


best_weights, best_score = None, -1.0
weight_step = 0.1
weight_candidates = [tuple(v * weight_step for v in ints) for ints in generate_weight_candidates(len(MODEL_NAMES), int(round(1.0 / weight_step)))]

for weights in weight_candidates:
    blend = sum(w * oof[m] for w, m in zip(weights, MODEL_NAMES))
    score = balanced_accuracy_score(y, blend.argmax(axis=1))
    if score > best_score:
        best_score = score
        best_weights = weights

print("n_weight_candidates:", len(weight_candidates))
print(f"best weights ({', '.join(MODEL_NAMES)}) = {best_weights}")
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
oof_pred_base = oof_blend_base.argmax(axis=1)

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


CELL13_EXP002 = """# OOF probabilities -> meta features
meta_features = np.concatenate([oof[m] for m in MODEL_NAMES], axis=1)
meta_test_features = np.concatenate([pred_test[m] for m in MODEL_NAMES], axis=1)

# meta CV
meta_oof = np.zeros((len(X), N_CLASSES))
meta_test = np.zeros((len(X_test), N_CLASSES))

meta_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + 999)
for fold, (tr_idx, va_idx) in enumerate(meta_skf.split(meta_features, y)):
    X_meta_tr, X_meta_va = meta_features[tr_idx], meta_features[va_idx]
    y_meta_tr, y_meta_va = y[tr_idx], y[va_idx]

    meta_model = LogisticRegression(
        max_iter=4000,
        multi_class="multinomial",
        class_weight="balanced",
        n_jobs=-1,
        C=2.0,
        random_state=SEED + 999 + fold,
    )
    meta_model.fit(X_meta_tr, y_meta_tr)

    meta_oof[va_idx] = meta_model.predict_proba(X_meta_va)
    meta_test += meta_model.predict_proba(meta_test_features) / N_FOLDS

meta_score = balanced_accuracy_score(y, meta_oof.argmax(axis=1))
print(f"stacking OOF balanced accuracy = {meta_score:.5f}")

test_blend = meta_test

cm_stack = confusion_matrix(y, meta_oof.argmax(axis=1), normalize="true")
disp_stack = ConfusionMatrixDisplay(cm_stack, display_labels=target_encoder.classes_)
disp_stack.plot(cmap="Blues", values_format=".3f")
plt.title("OOF confusion matrix (stacking)")
plt.show()

test_pred = target_encoder.inverse_transform(test_blend.argmax(axis=1))

submission = pd.DataFrame({"id": test_fe["id"], "class": test_pred})
assert (submission["id"].values == sample_sub["id"].values).all()
submission.to_csv("submission.csv", index=False)

print(submission["class"].value_counts())
submission.head()
"""


CELL13_EXP003 = """# OOF probabilities -> meta features
meta_features = np.concatenate([oof[m] for m in MODEL_NAMES], axis=1)
meta_test_features = np.concatenate([pred_test[m] for m in MODEL_NAMES], axis=1)

# meta CV with LightGBM
meta_oof = np.zeros((len(X), N_CLASSES))
meta_test = np.zeros((len(X_test), N_CLASSES))

meta_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + 999)
for fold, (tr_idx, va_idx) in enumerate(meta_skf.split(meta_features, y)):
    X_meta_tr, X_meta_va = meta_features[tr_idx], meta_features[va_idx]
    y_meta_tr, y_meta_va = y[tr_idx], y[va_idx]
    w_meta_tr = compute_sample_weight("balanced", y_meta_tr)

    meta_model = lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=SEED + 999 + fold,
        n_jobs=-1,
        verbosity=-1,
    )
    meta_model.fit(
        X_meta_tr,
        y_meta_tr,
        sample_weight=w_meta_tr,
        eval_set=[(X_meta_va, y_meta_va)],
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )

    meta_oof[va_idx] = meta_model.predict_proba(X_meta_va)
    meta_test += meta_model.predict_proba(meta_test_features) / N_FOLDS

meta_score = balanced_accuracy_score(y, meta_oof.argmax(axis=1))
print(f"stacking OOF balanced accuracy (meta=LGBM) = {meta_score:.5f}")

test_blend = meta_test

cm_stack = confusion_matrix(y, meta_oof.argmax(axis=1), normalize="true")
disp_stack = ConfusionMatrixDisplay(cm_stack, display_labels=target_encoder.classes_)
disp_stack.plot(cmap="Blues", values_format=".3f")
plt.title("OOF confusion matrix (stacking meta=LGBM)")
plt.show()

test_pred = target_encoder.inverse_transform(test_blend.argmax(axis=1))

submission = pd.DataFrame({"id": test_fe["id"], "class": test_pred})
assert (submission["id"].values == sample_sub["id"].values).all()
submission.to_csv("submission.csv", index=False)

print(submission["class"].value_counts())
submission.head()
"""


CELL13_EXP004 = """from itertools import product

# OOF probabilities -> meta features
meta_features = np.concatenate([oof[m] for m in MODEL_NAMES], axis=1)
meta_test_features = np.concatenate([pred_test[m] for m in MODEL_NAMES], axis=1)

# meta CV with LightGBM
meta_oof = np.zeros((len(X), N_CLASSES))
meta_test = np.zeros((len(X_test), N_CLASSES))

meta_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + 999)
for fold, (tr_idx, va_idx) in enumerate(meta_skf.split(meta_features, y)):
    X_meta_tr, X_meta_va = meta_features[tr_idx], meta_features[va_idx]
    y_meta_tr, y_meta_va = y[tr_idx], y[va_idx]
    w_meta_tr = compute_sample_weight("balanced", y_meta_tr)

    meta_model = lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=SEED + 999 + fold,
        n_jobs=-1,
        verbosity=-1,
    )
    meta_model.fit(
        X_meta_tr,
        y_meta_tr,
        sample_weight=w_meta_tr,
        eval_set=[(X_meta_va, y_meta_va)],
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )

    meta_oof[va_idx] = meta_model.predict_proba(X_meta_va)
    meta_test += meta_model.predict_proba(meta_test_features) / N_FOLDS


def apply_temperature(proba, temperature):
    logits = np.log(np.clip(proba, 1e-15, 1.0))
    scaled = logits / temperature
    exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)


def apply_class_thresholds(proba, class_thresholds):
    adj = proba / class_thresholds[np.newaxis, :]
    adj = np.clip(adj, 1e-15, None)
    return adj / adj.sum(axis=1, keepdims=True)


temp_grid = np.arange(0.70, 1.31, 0.05)
best_temp, best_temp_score = 1.0, -1.0
for t in temp_grid:
    p_tmp = apply_temperature(meta_oof, t)
    s = balanced_accuracy_score(y, p_tmp.argmax(axis=1))
    if s > best_temp_score:
        best_temp = float(t)
        best_temp_score = s

meta_oof_temp = apply_temperature(meta_oof, best_temp)
meta_test_temp = apply_temperature(meta_test, best_temp)
print(f"best temperature = {best_temp:.2f}, score = {best_temp_score:.5f}")

th_grid = [0.90, 0.95, 1.00, 1.05, 1.10]
best_th = np.ones(N_CLASSES)
best_th_score = -1.0
for th in product(th_grid, repeat=N_CLASSES):
    th = np.array(th)
    p_tmp = apply_class_thresholds(meta_oof_temp, th)
    s = balanced_accuracy_score(y, p_tmp.argmax(axis=1))
    if s > best_th_score:
        best_th = th.copy()
        best_th_score = s

meta_oof_cal = apply_class_thresholds(meta_oof_temp, best_th)
meta_test_cal = apply_class_thresholds(meta_test_temp, best_th)
print(f"best thresholds = {best_th.tolist()}, score = {best_th_score:.5f}")

meta_score_raw = balanced_accuracy_score(y, meta_oof.argmax(axis=1))
meta_score_cal = balanced_accuracy_score(y, meta_oof_cal.argmax(axis=1))
print(f"stacking OOF balanced accuracy (raw) = {meta_score_raw:.5f}")
print(f"stacking OOF balanced accuracy (temp+threshold) = {meta_score_cal:.5f}")

test_blend = meta_test_cal

cm_stack = confusion_matrix(y, meta_oof_cal.argmax(axis=1), normalize="true")
disp_stack = ConfusionMatrixDisplay(cm_stack, display_labels=target_encoder.classes_)
disp_stack.plot(cmap="Blues", values_format=".3f")
plt.title("OOF confusion matrix (stacking + temp/threshold)")
plt.show()

test_pred = target_encoder.inverse_transform(test_blend.argmax(axis=1))

submission = pd.DataFrame({"id": test_fe["id"], "class": test_pred})
assert (submission["id"].values == sample_sub["id"].values).all()
submission.to_csv("submission.csv", index=False)

print(submission["class"].value_counts())
submission.head()
"""


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_readme(path: str, title: str, bullets: list[str]) -> None:
    readme_path = os.path.join(path, "README.md")
    lines = [f"# {title}", "", "## 変更点"] + [f"- {b}" for b in bullets] + [""]
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_variant(out_dir: str, cell13_source: str, readme_title: str, readme_bullets: list[str]) -> None:
    nb = nbformat.read(BASE_NB, as_version=4)
    exp_name = os.path.basename(out_dir).replace("\\", "/")
    exp_suffix = "".join(ch for ch in exp_name if ch.isdigit())[-3:] or "000"
    cell1_source = CELL1_COMMON.replace(
        "from sklearn.linear_model import LogisticRegression",
        "from sklearn.linear_model import LogisticRegression\nfrom pathlib import Path",
    )
    cell1_source += (
        '\nEXP_OUTPUT_DIR = Path(".")\n'
        "EXP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
        f'SUBMISSION_FILENAME = "submission-{exp_suffix}.csv"\n'
    )
    nb.cells[1]["source"] = cell1_source
    nb.cells[8]["source"] = CELL8_SEED_AVG
    nb.cells[10]["source"] = CELL10_SEED_AVG
    nb.cells[13]["source"] = cell13_source.replace(
        'submission.to_csv("submission.csv", index=False)',
        'submission_path = EXP_OUTPUT_DIR / SUBMISSION_FILENAME\nsubmission.to_csv(submission_path, index=False)\nprint(f"saved: {submission_path}")',
    )

    ensure_dir(out_dir)
    out_nb_path = os.path.join(out_dir, "stellar_classification.ipynb")
    nbformat.write(nb, out_nb_path)
    write_readme(out_dir, readme_title, readme_bullets)


def main():
    exp002 = os.path.join(ROOT, "EXP", "EXP002")
    exp003 = os.path.join(ROOT, "EXP", "EXP003")
    exp004 = os.path.join(ROOT, "EXP", "EXP004")

    build_variant(
        exp002,
        CELL13_EXP002,
        "EXP002",
        [
            "5-fold CV + seed平均（SEED_LIST = [42, 52, 62]）を導入",
            "baseモデルは既存6モデルを継続",
            "メタ層はLogisticRegressionのまま",
        ],
    )
    build_variant(
        exp003,
        CELL13_EXP003,
        "EXP003",
        [
            "EXP002の設定をベースに維持",
            "メタ層をLogisticRegressionからLightGBMへ変更",
            "メタ層でもfold学習でリークを回避",
        ],
    )
    build_variant(
        exp004,
        CELL13_EXP004,
        "EXP004",
        [
            "EXP003（meta=LGBM）をベースに維持",
            "OOFでtemperature scalingを最適化",
            "クラス別thresholdをグリッド探索で最適化",
        ],
    )

    print("Created:")
    print(exp002)
    print(exp003)
    print(exp004)


if __name__ == "__main__":
    main()

