"""Check final report numerical claims against generated experiment outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DELIVERABLES = ROOT / "deliverables"


def fmt3(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    dataset = json.loads((RESULTS / "dataset_audit.json").read_text(encoding="utf-8"))
    notes = json.loads((RESULTS / "report_notes.json").read_text(encoding="utf-8"))
    models = pd.read_csv(RESULTS / "model_results_summary.csv")
    thresholds = pd.read_csv(RESULTS / "threshold_results.csv")
    main_tex = (ROOT / "main.tex").read_text(encoding="utf-8")

    checks: list[tuple[str, str, bool]] = []

    def add(name: str, value: str | int | float, expected: str | int | float) -> None:
        checks.append((name, str(value), str(expected) == str(value)))

    add("Dataset rows", dataset["shape"][0], 100000)
    add("Dataset variables", dataset["shape"][1], 11)
    add("Missing values total", sum(dataset["missing_values"].values()), 0)
    add("Depression=1 count", dataset["class_counts"]["True"], 10062)
    add("Depression=0 count", dataset["class_counts"]["False"], 89938)
    add("Train rows", dataset["train_size"], 80000)
    add("Test rows", dataset["test_size"], 20000)
    add("Train depressed count", dataset["train_class_counts"]["1"], 8050)
    add("Test depressed count", dataset["test_class_counts"]["1"], 2012)

    final = models.loc[models["model"] == "AdaBoost, threshold 0.3"].iloc[0]
    expected_final = {
        "accuracy": "0.731",
        "precision": "0.217",
        "recall": "0.644",
        "f1": "0.325",
        "balanced_accuracy": "0.692",
        "roc_auc": "0.697",
        "pr_auc": "0.226",
        "tn": "13322",
        "fp": "4666",
        "fn": "716",
        "tp": "1296",
    }
    for col, expected in expected_final.items():
        value = fmt3(final[col]) if col not in {"tn", "fp", "fn", "tp"} else str(int(final[col]))
        add(f"Final model {col}", value, expected)

    rf04 = thresholds[(thresholds["model"] == "Random Forest tuned") & (thresholds["threshold"] == 0.4)].iloc[0]
    rf05 = thresholds[(thresholds["model"] == "Random Forest tuned") & (thresholds["threshold"] == 0.5)].iloc[0]
    add("RF tuned threshold 0.4 recall", fmt3(rf04["recall"]), "0.645")
    add("RF tuned threshold 0.4 F1", fmt3(rf04["f1"]), "0.322")
    add("RF tuned threshold 0.4 balanced accuracy", fmt3(rf04["balanced_accuracy"]), "0.690")
    add("RF tuned threshold 0.4 false positives", int(rf04["fp"]), 4745)
    add("RF tuned threshold 0.4 false negatives", int(rf04["fn"]), 715)
    add("RF tuned threshold 0.5 recall", fmt3(rf05["recall"]), "0.629")
    add("RF tuned threshold 0.5 F1", fmt3(rf05["f1"]), "0.323")
    add("RF tuned threshold 0.5 balanced accuracy", fmt3(rf05["balanced_accuracy"]), "0.688")
    add("RF tuned threshold 0.5 false positives", int(rf05["fp"]), 4554)
    add("RF tuned threshold 0.5 false negatives", int(rf05["fn"]), 747)

    flags = int(final["tp"] + final["fp"])
    false_flag_rate = final["fp"] / flags
    add("Final positive flags", flags, 5962)
    add("False-positive share of flags", f"{false_flag_rate:.1%}", "78.3%")
    add("Selected model in report notes", notes["selected_model"], "AdaBoost")
    add("Selected threshold in report notes", notes["selected_threshold"], 0.3)

    required_snippets = [
        "AdaBoost at threshold 0.3",
        "0.731",
        "0.217",
        "0.644",
        "0.325",
        "0.692",
        "0.697",
        "0.226",
        "1,296",
        "716",
        "4,666",
        "13,322",
        "5,962",
        "78.3\\%",
        "4,745",
        "4,554",
        "unverified",
    ]
    for snippet in required_snippets:
        checks.append((f"LaTeX contains `{snippet}`", "present" if snippet in main_tex else "missing", snippet in main_tex))

    passed = sum(1 for _, _, ok in checks if ok)
    failed = len(checks) - passed
    lines = [
        "# Number Accuracy Audit",
        "",
        f"- Checks passed: {passed}",
        f"- Checks failed: {failed}",
        "",
        "| Check | Observed value | Status |",
        "|---|---:|---|",
    ]
    for name, value, ok in checks:
        lines.append(f"| {name} | {value} | {'PASS' if ok else 'FAIL'} |")
    lines.extend(
        [
            "",
            "Source files checked: `results/dataset_audit.json`, `results/model_results_summary.csv`, "
            "`results/threshold_results.csv`, `results/report_notes.json`, and `main.tex`.",
        ]
    )
    output = DELIVERABLES / "number_accuracy_audit.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if failed:
        raise SystemExit(f"Number audit failed with {failed} failed checks. See {output}.")
    print(f"Number audit passed. Wrote {output}.")


if __name__ == "__main__":
    main()
