from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "EXP"
OUT_DIR = Path(__file__).resolve().parent

# Edit this table to try different blends quickly.
BLEND_SOURCES = [
    ("EXP003", EXP_DIR / "EXP003" / "submission.csv", 0.60),
    ("EXP004", EXP_DIR / "EXP004" / "submission.csv", 0.10),
    ("EXP005", EXP_DIR / "EXP005" / "submission-005.csv", 0.30),
]

OUTPUT_NAME = "submission-006.csv"


def load_submission(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing submission file: {path}")
    df = pd.read_csv(path)
    if "id" not in df.columns or "class" not in df.columns:
        raise ValueError(f"Invalid format (need id,class): {path}")
    return df[["id", "class"]].copy()


def weighted_vote(row: pd.Series, model_names: list[str], model_weights: dict[str, float], class_order: list[str]) -> str:
    scores = {c: 0.0 for c in class_order}
    for model in model_names:
        pred = row[model]
        scores[pred] = scores.get(pred, 0.0) + model_weights[model]
    # deterministic tie-break with class_order
    return max(class_order, key=lambda c: (scores.get(c, 0.0), -class_order.index(c)))


def main() -> None:
    class_order = ["GALAXY", "QSO", "STAR"]

    model_names: list[str] = []
    model_weights: dict[str, float] = {}
    merged = None

    for model_name, file_path, weight in BLEND_SOURCES:
        model_names.append(model_name)
        model_weights[model_name] = float(weight)
        df = load_submission(file_path).rename(columns={"class": model_name})
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="id", how="inner")

    if merged is None:
        raise RuntimeError("No blend sources configured.")

    # basic safety check: all ids should be present after joins
    expected_rows = len(load_submission(BLEND_SOURCES[0][1]))
    if len(merged) != expected_rows:
        raise ValueError(f"ID mismatch across submissions: {len(merged)} vs expected {expected_rows}")

    merged["class"] = merged.apply(
        lambda r: weighted_vote(r, model_names, model_weights, class_order), axis=1
    )

    out = merged[["id", "class"]].copy()
    out_path = OUT_DIR / OUTPUT_NAME
    out.to_csv(out_path, index=False)

    print("saved:", out_path)
    print("weights:", model_weights)
    print(out["class"].value_counts())


if __name__ == "__main__":
    main()

