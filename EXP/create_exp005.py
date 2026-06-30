import os
import nbformat

ROOT = r"D:/program/Kaggle/Kaggle-Predicting-Stellar-Class"
SRC = os.path.join(ROOT, "EXP", "EXP003", "stellar_classification.ipynb")
OUT_DIR = os.path.join(ROOT, "EXP", "EXP005")
OUT_NB = os.path.join(OUT_DIR, "stellar_classification.ipynb")

os.makedirs(OUT_DIR, exist_ok=True)

nb = nbformat.read(SRC, as_version=4)

nb.cells[1]["source"] = nb.cells[1]["source"].replace(
    'SUBMISSION_FILENAME = "submission-003.csv"',
    'SUBMISSION_FILENAME = "submission-005.csv"',
)

nb.cells[13]["source"] = """# OOF probabilities -> meta features
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

meta_score_raw = balanced_accuracy_score(y, meta_oof.argmax(axis=1))
print(f"stacking OOF balanced accuracy (meta=LGBM, raw) = {meta_score_raw:.5f}")

# STAR-first tuning (EXP005-lite): boost STAR posterior only
star_idx = int(np.where(target_encoder.classes_ == "STAR")[0][0])
star_grid = np.arange(1.00, 1.51, 0.05)
best_star_w, best_star_score = 1.0, -1.0
for ws in star_grid:
    w = np.ones(N_CLASSES)
    w[star_idx] = ws
    pred_tmp = (meta_oof * w).argmax(axis=1)
    s = balanced_accuracy_score(y, pred_tmp)
    if s > best_star_score:
        best_star_score = s
        best_star_w = float(ws)

star_w = np.ones(N_CLASSES)
star_w[star_idx] = best_star_w
meta_oof_star = meta_oof * star_w
meta_test_star = meta_test * star_w
meta_oof_star = meta_oof_star / meta_oof_star.sum(axis=1, keepdims=True)
meta_test_star = meta_test_star / meta_test_star.sum(axis=1, keepdims=True)

meta_score_star = balanced_accuracy_score(y, meta_oof_star.argmax(axis=1))
print(f"best STAR weight = {best_star_w:.2f}")
print(f"stacking OOF balanced accuracy (STAR-tuned) = {meta_score_star:.5f}")

# Use STAR-tuned output for submission
test_blend = meta_test_star

cm_stack = confusion_matrix(y, meta_oof_star.argmax(axis=1), normalize="true")
disp_stack = ConfusionMatrixDisplay(cm_stack, display_labels=target_encoder.classes_)
disp_stack.plot(cmap="Blues", values_format=".3f")
plt.title("OOF confusion matrix (stacking meta=LGBM + STAR-tuned)")
plt.show()

test_pred = target_encoder.inverse_transform(test_blend.argmax(axis=1))

submission = pd.DataFrame({"id": test_fe["id"], "class": test_pred})
assert (submission["id"].values == sample_sub["id"].values).all()
submission_path = EXP_OUTPUT_DIR / SUBMISSION_FILENAME
submission.to_csv(submission_path, index=False)
print(f"saved: {submission_path}")

print(submission["class"].value_counts())
submission.head()
"""

nbformat.write(nb, OUT_NB)

with open(os.path.join(OUT_DIR, "README.md"), "w", encoding="utf-8") as f:
    f.write(
        "# EXP005\\n\\n"
        "## 変更点\\n"
        "- EXP003(meta=LGBM)をベース\\n"
        "- STARクラスのみ後段で重み最適化（OOFで探索）\\n"
        "- 出力は submission-005.csv\\n"
    )

print("created", OUT_NB)
