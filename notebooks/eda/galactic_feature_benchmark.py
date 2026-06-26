import time
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight


SEED = 42
N_FOLDS = 3


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["u_g"] = out["u"] - out["g"]
    out["g_r"] = out["g"] - out["r"]
    out["r_i"] = out["r"] - out["i"]
    out["i_z"] = out["i"] - out["z"]
    out["u_r"] = out["u"] - out["r"]
    out["g_i"] = out["g"] - out["i"]
    out["u_z"] = out["u"] - out["z"]
    out["redshift_log1p"] = np.log1p(out["redshift"].clip(lower=0))
    return out


def equatorial_to_galactic(alpha_deg: np.ndarray, delta_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # J2000 constants
    ra_ngp = np.deg2rad(192.85948)
    dec_ngp = np.deg2rad(27.12825)
    l_omega = np.deg2rad(32.93192)

    ra = np.deg2rad(alpha_deg)
    dec = np.deg2rad(delta_deg)

    sin_b = np.sin(dec) * np.sin(dec_ngp) + np.cos(dec) * np.cos(dec_ngp) * np.cos(ra - ra_ngp)
    b = np.arcsin(np.clip(sin_b, -1.0, 1.0))

    y = np.cos(dec) * np.sin(ra - ra_ngp)
    x = np.sin(dec) * np.cos(dec_ngp) - np.cos(dec) * np.sin(dec_ngp) * np.cos(ra - ra_ngp)
    l = np.arctan2(y, x) + l_omega
    l = np.mod(l, 2.0 * np.pi)

    return np.rad2deg(l), np.rad2deg(b)


def add_galactic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    l_deg, b_deg = equatorial_to_galactic(out["alpha"].values, out["delta"].values)
    out["l"] = l_deg
    out["b"] = b_deg
    return out


def add_trig_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    rad = np.deg2rad(out[col].values)
    out[f"{col}_sin"] = np.sin(rad)
    out[f"{col}_cos"] = np.cos(rad)
    return out


def preprocess(df: pd.DataFrame, cat_cols: list[str], encoders: dict[str, LabelEncoder] | None = None):
    out = add_base_features(df)
    if encoders is None:
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            out[col] = le.fit_transform(out[col].astype(str))
            encoders[col] = le
    else:
        for col in cat_cols:
            out[col] = encoders[col].transform(out[col].astype(str))
    return out, encoders


def evaluate_variant(name: str, X_df: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray) -> tuple[str, float, float]:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X_df), dtype=np.int64)
    start = time.time()

    for tr_idx, va_idx in skf.split(X_df, y):
        X_tr, X_va = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        w_tr = sample_weight[tr_idx]

        model = LGBMClassifier(
            objective="multiclass",
            n_estimators=1800,
            learning_rate=0.03,
            num_leaves=127,
            max_depth=-1,
            min_child_samples=30,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1,
        )
        model.fit(
            X_tr,
            y_tr,
            sample_weight=w_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="multi_logloss",
            callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
        )
        oof[va_idx] = model.predict(X_va)

    score = balanced_accuracy_score(y, oof)
    elapsed = time.time() - start
    return name, score, elapsed


def main():
    train = pd.read_csv("data/train.csv")
    cat_cols = ["spectral_type", "galaxy_population"]

    train_fe, cat_encoders = preprocess(train, cat_cols)

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(train_fe["class"])
    sample_weight = compute_sample_weight("balanced", y)

    common_drop = {"id", "class"}

    variants: dict[str, pd.DataFrame] = {}

    # Baseline
    variants["baseline"] = train_fe.copy()

    # + galactic coordinates
    v_lb = add_galactic_features(train_fe)
    variants["+l_b"] = v_lb

    # + galactic trig
    v_lb_trig = add_trig_features(add_trig_features(v_lb, "l"), "b")
    variants["+l_b+trig_lb"] = v_lb_trig

    # + equatorial trig
    v_eq_trig = add_trig_features(add_trig_features(train_fe, "alpha"), "delta")
    variants["+trig_alpha_delta"] = v_eq_trig

    # full
    v_full = add_trig_features(add_trig_features(v_lb_trig, "alpha"), "delta")
    variants["+l_b+trig_lb+trig_alpha_delta"] = v_full

    # use only galactic representation for direction (drop alpha/delta)
    v_gal_only = v_lb_trig.drop(columns=["alpha", "delta"])
    variants["+l_b+trig_lb(drop_alpha_delta)"] = v_gal_only

    results = []
    for name, dfv in variants.items():
        features = [c for c in dfv.columns if c not in common_drop]
        X = dfv[features]
        print(f"\n[{name}] n_features={len(features)}")
        r = evaluate_variant(name, X, y, sample_weight)
        print(f"score={r[1]:.5f}, elapsed={r[2]:.1f}s")
        results.append(r)

    res_df = pd.DataFrame(results, columns=["variant", "balanced_accuracy", "elapsed_sec"])
    res_df = res_df.sort_values("balanced_accuracy", ascending=False).reset_index(drop=True)

    print("\n=== Summary (best first) ===")
    print(res_df.to_string(index=False))

    out_path = "notebooks/eda/galactic_feature_benchmark_results.csv"
    res_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

