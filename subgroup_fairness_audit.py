"""Subgroup error audit for the selected AdaBoost operating point.

This script retrains the final documented pipeline on the same stratified split
and writes subgroup metrics for the held-out test set. It asserts the published
confusion matrix before saving results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from run_experiments import RANDOM_STATE, build_pipeline, make_preprocessor


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "student_lifestyle_100k.csv"
RESULTS_PATH = ROOT / "results" / "subgroup_fairness_results.csv"


def rate(num: int, den: int) -> float:
    return float(num / den) if den else np.nan


def subgroup_rows(frame: pd.DataFrame, column: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for value, group in frame.groupby(column, dropna=False, observed=True):
        if group.empty:
            continue
        tn, fp, fn, tp = confusion_matrix(group["y_true"], group["y_pred"], labels=[0, 1]).ravel()
        rows.append(
            {
                "subgroup_variable": column,
                "subgroup": str(value),
                "support": int(len(group)),
                "positive_support": int((group["y_true"] == 1).sum()),
                "negative_support": int((group["y_true"] == 0).sum()),
                "precision": rate(int(tp), int(tp + fp)),
                "recall": rate(int(tp), int(tp + fn)),
                "false_positive_rate": rate(int(fp), int(fp + tn)),
                "false_negative_rate": rate(int(fn), int(fn + tp)),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    return rows


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    y = df["Depression"].astype(int)
    X = df.drop(columns=["Student_ID", "Depression"])

    numeric_features = [
        "Age",
        "CGPA",
        "Sleep_Duration",
        "Study_Hours",
        "Social_Media_Hours",
        "Physical_Activity",
        "Stress_Level",
    ]
    categorical_features = ["Gender", "Department"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    model = build_pipeline(
        make_preprocessor(numeric_features, categorical_features),
        AdaBoostClassifier(random_state=RANDOM_STATE, n_estimators=120),
    )
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    y_pred = (scores >= 0.3).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    expected = (13322, 4666, 716, 1296)
    observed = (int(tn), int(fp), int(fn), int(tp))
    if observed != expected:
        raise RuntimeError(f"Final confusion matrix mismatch: observed {observed}, expected {expected}")

    audit = X_test.copy()
    audit["y_true"] = y_test.to_numpy()
    audit["y_pred"] = y_pred
    audit["Age band"] = pd.cut(
        audit["Age"],
        bins=[0, 20, 24, 200],
        labels=["<=20", "21-24", "25+"],
        include_lowest=True,
    )

    rows: list[dict[str, object]] = []
    for column in ["Gender", "Department", "Age band"]:
        rows.extend(subgroup_rows(audit, column))

    out = pd.DataFrame(rows)
    out = out.sort_values(["subgroup_variable", "subgroup"]).reset_index(drop=True)
    out.to_csv(RESULTS_PATH, index=False)
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
