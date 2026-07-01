from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "EXP"
OUT_DIR = Path(__file__).resolve().parent

# Best LB submission (0.96841). No retraining — label post-processing only.
BASE_SUBMISSION = EXP_DIR / "EXP007" / "submission-007.csv"
TEST_CSV = ROOT / "data" / "test.csv"
OUTPUT_NAME = "submission-009.csv"

# Rules: (name, source_class, target_class, mask on merged df)
# Only high-purity rules enabled by default (train z>1.5 → QSO 98.7%).
RULES = [
    {
        "name": "galaxy_to_qso_high_z",
        "from_class": "GALAXY",
        "to_class": "QSO",
        "mask": lambda df: df["redshift"] > 1.5,
    },
]

# Optional (disabled): low support in train or large flip count
OPTIONAL_RULES = [
    {
        "name": "galaxy_to_qso_very_high_z",
        "from_class": "GALAXY",
        "to_class": "QSO",
        "mask": lambda df: df["redshift"] > 2.0,
    },
    {
        "name": "qso_to_galaxy_low_z",
        "from_class": "QSO",
        "to_class": "GALAXY",
        "mask": lambda df: df["redshift"] < 0.3,
    },
]


def apply_rules(df: pd.DataFrame, rules: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    log_rows: list[dict] = []

    for rule in rules:
        mask = (
            (out["class"] == rule["from_class"])
            & rule["mask"](out)
        )
        n = int(mask.sum())
        if n == 0:
            continue
        for idx in out.index[mask]:
            log_rows.append(
                {
                    "id": int(out.at[idx, "id"]),
                    "rule": rule["name"],
                    "from": rule["from_class"],
                    "to": rule["to_class"],
                    "redshift": float(out.at[idx, "redshift"]),
                }
            )
        out.loc[mask, "class"] = rule["to_class"]

    return out, pd.DataFrame(log_rows)


def main() -> None:
    if not BASE_SUBMISSION.exists():
        raise FileNotFoundError(f"Missing base submission: {BASE_SUBMISSION}")
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Missing test data: {TEST_CSV}")

    base = pd.read_csv(BASE_SUBMISSION)
    test = pd.read_csv(TEST_CSV, usecols=["id", "redshift"])
    merged = test.merge(base, on="id", how="inner")
    if len(merged) != len(base):
        raise ValueError(f"ID mismatch: base={len(base)}, merged={len(merged)}")

    print("base submission:", BASE_SUBMISSION.name)
    print("base class counts:\n", base["class"].value_counts().to_string())
    print()

    out, log = apply_rules(merged, RULES)
    submission = out[["id", "class"]].copy()

    submission_path = OUT_DIR / OUTPUT_NAME
    submission.to_csv(submission_path, index=False)

    log_path = OUT_DIR / "redshift_postprocess_log.csv"
    if len(log):
        log.to_csv(log_path, index=False)
        print(f"flipped rows: {len(log)}")
        print(log.groupby(["rule", "from", "to"]).size().to_string())
        print(f"log: {log_path}")
    else:
        print("flipped rows: 0")

    changed = (submission["class"].values != base["class"].values).sum()
    print(f"\nchanged vs base: {changed} ({changed / len(submission) * 100:.3f}%)")
    print("saved:", submission_path)
    print("output class counts:\n", submission["class"].value_counts().to_string())


if __name__ == "__main__":
    main()
