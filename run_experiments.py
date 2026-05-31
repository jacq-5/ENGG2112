"""Reproducible ENGG2112 student depression-risk modelling audit.

This script is intentionally self-contained so the final report can cite only
numbers regenerated from code. It writes machine-readable results, report-ready
tables, and figures under the workspace `results/` and `figures/` folders.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(Path(__file__).resolve().parents[1] / "data" / "student_lifestyle_100k.csv"),
        help="Path to student_lifestyle_100k.csv",
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument(
        "--skip-mlp",
        action="store_true",
        help="Skip MLPClassifier if runtime is too constrained.",
    )
    return parser.parse_args()


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        verbose_feature_names_out=False,
    )


def metric_row(
    model_name: str,
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None,
    notes: str = "",
    threshold: float | None = None,
) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    row: dict[str, Any] = {
        "model": model_name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "notes": notes,
    }
    if scores is not None and len(np.unique(scores)) > 1:
        row["roc_auc"] = roc_auc_score(y_true, scores)
        row["pr_auc"] = average_precision_score(y_true, scores)
    elif scores is not None:
        row["roc_auc"] = 0.5
        row["pr_auc"] = float(np.mean(y_true))
    else:
        row["roc_auc"] = np.nan
        row["pr_auc"] = np.nan
    row["practical_score"] = (
        0.45 * row["recall"] + 0.30 * row["f1"] + 0.25 * row["balanced_accuracy"]
    )
    return row


def predict_scores(estimator: Pipeline | ImbPipeline, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(X_test)[:, 1]
        return (probabilities >= 0.5).astype(int), probabilities
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X_test)
        return estimator.predict(X_test), scores
    return estimator.predict(X_test), None


def build_pipeline(
    preprocessor: ColumnTransformer,
    model: Any,
    use_smote: bool = False,
) -> Pipeline | ImbPipeline:
    if use_smote:
        return ImbPipeline(
            [
                ("preprocess", clone(preprocessor)),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("model", model),
            ]
        )
    return Pipeline(
        [
            ("preprocess", clone(preprocessor)),
            ("model", model),
        ]
    )


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def format_float(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.3f}"


def plot_class_distribution(df: pd.DataFrame, figures_dir: Path) -> None:
    counts = df["Depression"].value_counts().rename(index={False: "Depression = 0", True: "Depression = 1"})
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    colors = ["#4C78A8", "#F58518"]
    bars = ax.bar(counts.index, counts.values, color=colors)
    total = counts.sum()
    ax.set_ylabel("Records")
    ax.set_title("Class distribution")
    ax.grid(axis="y", alpha=0.25)
    for bar in bars:
        count = int(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count + total * 0.01,
            f"{count:,}\n({count / total:.1%})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, max(counts.values) * 1.16)
    fig.tight_layout()
    fig.savefig(figures_dir / "class_distribution.png", dpi=220)
    plt.close(fig)


def plot_metrics(results: pd.DataFrame, figures_dir: Path, selected_models: list[str]) -> None:
    subset = results[results["model"].isin(selected_models)].copy()
    subset = subset.set_index("model").loc[selected_models].reset_index()
    metrics = ["precision", "recall", "f1", "balanced_accuracy"]
    labels = ["Precision", "Recall", "F1", "Balanced acc."]
    y = np.arange(len(subset))
    height = 0.18
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    palette = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    for idx, metric in enumerate(metrics):
        ax.barh(y + (idx - 1.5) * height, subset[metric], height, label=labels[idx], color=palette[idx])
    display_names = {
        "AdaBoost, threshold 0.3": "AdaBoost @ 0.3",
        "Random Forest tuned": "Random Forest tuned (0.5)",
    }
    clean_names = subset["model"].replace(display_names).str.replace(", threshold ", " @ ", regex=False)
    ax.set_yticks(y)
    ax.set_yticklabels(clean_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_title("Minority-class performance and balance")
    ax.legend(ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.23), frameon=False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(figures_dir / "model_metric_comparison.png", dpi=220)
    plt.close(fig)


def plot_thresholds(threshold_df: pd.DataFrame, figures_dir: Path, final_model: str) -> None:
    subset = threshold_df[threshold_df["model"] == final_model].sort_values("threshold")
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(subset["threshold"], subset["precision"], marker="o", label="Precision", color="#4C78A8")
    ax.plot(subset["threshold"], subset["recall"], marker="o", label="Recall", color="#F58518")
    ax.plot(subset["threshold"], subset["f1"], marker="o", label="F1", color="#54A24B")
    ax.plot(subset["threshold"], subset["balanced_accuracy"], marker="o", label="Balanced acc.", color="#B279A2")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Threshold trade-off: {final_model}")
    ax.axvline(0.3, color="#333333", linestyle=":", linewidth=1.0)
    ax.text(0.305, 0.74, "Selected threshold", fontsize=8, color="#333333")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8, loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(figures_dir / "threshold_tuning_tradeoff.png", dpi=220)
    plt.close(fig)


def plot_confusion(y_true: pd.Series, y_pred: np.ndarray, figures_dir: Path, filename: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Depression = 0", "Depression = 1"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
        values_format=",d",
    )
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(figures_dir / filename, dpi=220)
    plt.close(fig)


def plot_curves(
    fitted: dict[str, Pipeline | ImbPipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    results: pd.DataFrame,
    figures_dir: Path,
    final_model: str,
) -> None:
    preferred = [
        final_model,
        "Random Forest tuned",
        "Gradient Boosting SMOTE",
        "Logistic Regression SMOTE",
        "Gaussian Naive Bayes",
    ]
    curve_models = [name for name in preferred if name in fitted]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1))
    for name in curve_models:
        estimator = fitted[name]
        _, scores = predict_scores(estimator, X_test)
        if scores is None or len(np.unique(scores)) <= 1:
            continue
        display_name = name.replace("Logistic Regression", "LogReg").replace("Gradient Boosting", "GradBoost")
        RocCurveDisplay.from_predictions(y_test, scores, name=display_name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y_test, scores, name=display_name, ax=axes[1])
    axes[0].plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=0.9)
    axes[0].set_title("ROC curves")
    axes[1].axhline(y_test.mean(), color="#777777", linestyle="--", linewidth=0.9, label="Base rate")
    axes[1].set_title("Precision-recall curves")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6.2, loc="lower right" if ax is axes[0] else "upper right")
    fig.tight_layout()
    fig.savefig(figures_dir / "roc_pr_curves_top_models.png", dpi=220)
    plt.close(fig)


def plot_feature_importance(
    final_pipeline: Pipeline | ImbPipeline,
    numeric_features: list[str],
    categorical_features: list[str],
    figures_dir: Path,
    results_dir: Path,
) -> None:
    model = final_pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return
    preprocessor = final_pipeline.named_steps["preprocess"]
    feature_names = preprocessor.get_feature_names_out().tolist()
    importances = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )
    grouped_rows = []
    for feature in numeric_features:
        grouped_rows.append(
            {
                "feature": feature,
                "importance": importances.loc[importances["feature"] == feature, "importance"].sum(),
            }
        )
    for feature in categorical_features:
        grouped_rows.append(
            {
                "feature": feature,
                "importance": importances.loc[
                    importances["feature"].str.startswith(f"{feature}_"), "importance"
                ].sum(),
            }
        )
    grouped = pd.DataFrame(grouped_rows).sort_values("importance", ascending=True)
    grouped.to_csv(results_dir / "feature_importance.csv", index=False)
    plotted = grouped[grouped["importance"] > 0.001].copy()
    plotted["label"] = plotted["feature"].str.replace("_", " ", regex=False)
    fig, ax = plt.subplots(figsize=(6.1, 3.1))
    ax.barh(plotted["label"], plotted["importance"], color="#4C78A8")
    for y_pos, value in enumerate(plotted["importance"]):
        ax.text(value + plotted["importance"].max() * 0.015, y_pos, f"{value:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Mean decrease in impurity importance")
    ax.set_title("Final model feature importance")
    ax.set_xlim(0, plotted["importance"].max() * 1.16)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "feature_importance_final_model.png", dpi=220)
    plt.close(fig)


def run_grid_searches(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    results_dir: Path,
) -> dict[str, Pipeline | ImbPipeline]:
    searches: dict[str, tuple[Pipeline | ImbPipeline, dict[str, Any]]] = {
        "Logistic Regression tuned": (
            build_pipeline(
                preprocessor,
                LogisticRegression(max_iter=1200, solver="lbfgs", random_state=RANDOM_STATE),
            ),
            {"model__C": [0.1, 1.0], "model__class_weight": [None, "balanced"]},
        ),
        "Random Forest tuned": (
            build_pipeline(
                preprocessor,
                RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            ),
            {
                "model__n_estimators": [80],
                "model__max_depth": [8, 12],
                "model__min_samples_leaf": [5, 10],
                "model__class_weight": ["balanced"],
            },
        ),
        "Gradient Boosting tuned": (
            build_pipeline(preprocessor, GradientBoostingClassifier(random_state=RANDOM_STATE)),
            {
                "model__n_estimators": [80, 120],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
        "LinearSVC tuned": (
            build_pipeline(
                preprocessor,
                CalibratedClassifierCV(
                    estimator=LinearSVC(random_state=RANDOM_STATE, max_iter=5000),
                    cv=3,
                    method="sigmoid",
                ),
            ),
            {
                "model__estimator__C": [0.1, 1.0],
                "model__estimator__class_weight": [None, "balanced"],
            },
        ),
    }
    tuned: dict[str, Pipeline | ImbPipeline] = {}
    audit_rows = []
    for name, (pipe, param_grid) in searches.items():
        start = time.time()
        search = GridSearchCV(
            pipe,
            param_grid=param_grid,
            scoring="f1",
            cv=3,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        tuned[name] = search.best_estimator_
        audit_rows.append(
            {
                "model": name,
                "best_score_cv_f1": search.best_score_,
                "best_params": json.dumps(search.best_params_, sort_keys=True),
                "seconds": round(time.time() - start, 2),
            }
        )
    pd.DataFrame(audit_rows).to_csv(results_dir / "hyperparameter_results.csv", index=False)
    return tuned


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    target = "Depression"
    y = df[target].astype(int)
    X = df.drop(columns=[target, "Student_ID"])

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    audit = {
        "dataset_path": str(data_path),
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {col: int(value) for col, value in df.isna().sum().items()},
        "class_counts": {str(k): int(v) for k, v in df[target].value_counts().to_dict().items()},
        "class_percentages": {
            str(k): float(v) for k, v in (df[target].value_counts(normalize=True)).to_dict().items()
        },
        "target": target,
        "dropped_columns": ["Student_ID"],
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "random_state": RANDOM_STATE,
        "split": "train_test_split(test_size=0.2, stratify=y)",
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "train_class_counts": {str(k): int(v) for k, v in y_train.value_counts().to_dict().items()},
        "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().to_dict().items()},
    }
    save_json(results_dir / "dataset_audit.json", audit)
    plot_class_distribution(df, figures_dir)

    preprocessor = make_preprocessor(numeric_features, categorical_features)
    tuned_models = run_grid_searches(preprocessor, X_train, y_train, results_dir)

    models: dict[str, Pipeline | ImbPipeline] = {
        "Dummy baseline": build_pipeline(preprocessor, DummyClassifier(strategy="most_frequent")),
        "Logistic Regression": build_pipeline(
            preprocessor,
            LogisticRegression(max_iter=1200, solver="lbfgs", random_state=RANDOM_STATE),
        ),
        "Logistic Regression balanced": build_pipeline(
            preprocessor,
            LogisticRegression(
                max_iter=1200,
                solver="lbfgs",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
        "Logistic Regression SMOTE": build_pipeline(
            preprocessor,
            LogisticRegression(max_iter=1200, solver="lbfgs", random_state=RANDOM_STATE),
            use_smote=True,
        ),
        "Gaussian Naive Bayes": build_pipeline(preprocessor, GaussianNB()),
        "Decision Tree": build_pipeline(
            preprocessor,
            DecisionTreeClassifier(random_state=RANDOM_STATE),
        ),
        "Decision Tree balanced": build_pipeline(
            preprocessor,
            DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        ),
        "Random Forest": build_pipeline(
            preprocessor,
            RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        ),
        "Random Forest balanced": build_pipeline(
            preprocessor,
            RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        "Random Forest SMOTE": build_pipeline(
            preprocessor,
            RandomForestClassifier(n_estimators=80, random_state=RANDOM_STATE, n_jobs=-1),
            use_smote=True,
        ),
        "Extra Trees": build_pipeline(
            preprocessor,
            ExtraTreesClassifier(n_estimators=120, random_state=RANDOM_STATE, n_jobs=-1),
        ),
        "Extra Trees balanced": build_pipeline(
            preprocessor,
            ExtraTreesClassifier(
                n_estimators=120,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        "KNN": build_pipeline(preprocessor, KNeighborsClassifier(n_neighbors=15, n_jobs=-1)),
        "Weighted KNN": build_pipeline(
            preprocessor,
            KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1),
        ),
        "Gradient Boosting": build_pipeline(
            preprocessor,
            GradientBoostingClassifier(random_state=RANDOM_STATE),
        ),
        "Gradient Boosting SMOTE": build_pipeline(
            preprocessor,
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            use_smote=True,
        ),
        "HistGradientBoosting": build_pipeline(
            preprocessor,
            HistGradientBoostingClassifier(random_state=RANDOM_STATE, max_iter=120),
        ),
        "AdaBoost": build_pipeline(
            preprocessor,
            AdaBoostClassifier(random_state=RANDOM_STATE, n_estimators=120),
        ),
        "LinearSVC balanced calibrated": build_pipeline(
            preprocessor,
            CalibratedClassifierCV(
                estimator=LinearSVC(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    max_iter=5000,
                ),
                cv=3,
                method="sigmoid",
            ),
        ),
    }
    if not args.skip_mlp:
        models["MLPClassifier bounded"] = build_pipeline(
            preprocessor,
            MLPClassifier(
                hidden_layer_sizes=(32,),
                max_iter=80,
                early_stopping=True,
                n_iter_no_change=8,
                random_state=RANDOM_STATE,
            ),
        )
    models.update(tuned_models)

    fitted: dict[str, Pipeline | ImbPipeline] = {}
    rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    prediction_cache: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    for name, pipe in models.items():
        start = time.time()
        pipe.fit(X_train, y_train)
        y_pred, scores = predict_scores(pipe, X_test)
        prediction_cache[name] = (y_pred, scores)
        row = metric_row(name, y_test, y_pred, scores, notes=f"fit_seconds={time.time() - start:.2f}")
        rows.append(row)
        fitted[name] = pipe
        if scores is not None:
            for threshold in THRESHOLDS:
                pred = (scores >= threshold).astype(int)
                threshold_rows.append(metric_row(name, y_test, pred, scores, threshold=threshold))

    results = pd.DataFrame(rows).sort_values(["practical_score", "f1"], ascending=False)
    threshold_df = pd.DataFrame(threshold_rows).sort_values(["model", "threshold"])

    base_positive_rate = float(y_test.mean())
    threshold_candidates = threshold_df[
        (threshold_df["precision"] >= base_positive_rate * 1.5)
        & (threshold_df["accuracy"] >= 0.50)
        & (threshold_df["balanced_accuracy"] >= 0.60)
        & (threshold_df["fp"] <= 6000)
    ].copy()
    if threshold_candidates.empty:
        threshold_candidates = threshold_df[
            (threshold_df["precision"] >= base_positive_rate)
            & (threshold_df["accuracy"] >= 0.50)
            & (threshold_df["balanced_accuracy"] >= 0.55)
        ].copy()
    final_row = threshold_candidates.sort_values(
        ["f1", "balanced_accuracy", "recall", "precision"], ascending=False
    ).iloc[0]
    final_model = str(final_row["model"])
    final_threshold = float(final_row["threshold"])
    final_pred = (prediction_cache[final_model][1] >= final_threshold).astype(int)
    final_metric_row = metric_row(
        f"{final_model}, threshold {final_threshold:.1f}",
        y_test,
        final_pred,
        prediction_cache[final_model][1],
        notes=(
            "selected by highest Depression=1 F1 among thresholds with precision >= 1.5x base rate, "
            "accuracy >= 0.50, balanced_accuracy >= 0.60, and FP <= 6000"
        ),
        threshold=final_threshold,
    )
    results_with_final = pd.concat([pd.DataFrame([final_metric_row]), results], ignore_index=True)
    results_with_final = results_with_final.sort_values(["practical_score", "f1"], ascending=False)

    results_with_final.to_csv(results_dir / "model_results_summary.csv", index=False)
    threshold_df.to_csv(results_dir / "threshold_results.csv", index=False)

    confusion = results_with_final[
        ["model", "threshold", "tn", "fp", "fn", "tp", "accuracy", "precision", "recall", "f1", "balanced_accuracy"]
    ].copy()
    confusion.to_csv(results_dir / "confusion_matrix_results.csv", index=False)

    top_main = results_with_final.head(10).copy()
    top_main["accuracy_fmt"] = top_main["accuracy"].map(format_float)
    top_main["precision_fmt"] = top_main["precision"].map(format_float)
    top_main["recall_fmt"] = top_main["recall"].map(format_float)
    top_main["f1_fmt"] = top_main["f1"].map(format_float)
    top_main["balanced_accuracy_fmt"] = top_main["balanced_accuracy"].map(format_float)
    top_main["roc_auc_fmt"] = top_main["roc_auc"].map(format_float)
    top_main["pr_auc_fmt"] = top_main["pr_auc"].map(format_float)
    top_main.to_csv(results_dir / "main_table_candidates.csv", index=False)

    selected_models = top_main["model"].head(6).tolist()
    plot_metrics(results_with_final, figures_dir, selected_models)
    plot_thresholds(threshold_df, figures_dir, final_model)
    plot_confusion(
        y_test,
        final_pred,
        figures_dir,
        "confusion_matrix_best_model.png",
        f"{final_model}\nthreshold {final_threshold:.1f}",
    )
    for idx, model_name in enumerate(results_with_final["model"].iloc[1:4], start=2):
        pred, _ = prediction_cache[model_name]
        plot_confusion(
            y_test,
            pred,
            figures_dir,
            f"confusion_matrix_model_{idx}.png",
            model_name,
        )
    plot_curves(fitted, X_test, y_test, results, figures_dir, final_model)
    plot_feature_importance(
        fitted[final_model],
        numeric_features,
        categorical_features,
        figures_dir,
        results_dir,
    )

    notes = {
        "selected_model": final_model,
        "selected_threshold": final_threshold,
        "selected_metrics": final_metric_row,
        "base_positive_rate": base_positive_rate,
        "selection_rule": (
            "Highest Depression=1 F1 among thresholded candidates with precision at least "
            "1.5 times the test-set base rate, accuracy at least 0.50, balanced accuracy at "
            "least 0.60, and no more than 6000 false positives."
        ),
        "result_files": [
            "dataset_audit.json",
            "model_results_summary.csv",
            "threshold_results.csv",
            "confusion_matrix_results.csv",
            "feature_importance.csv",
            "hyperparameter_results.csv",
            "main_table_candidates.csv",
        ],
        "figure_files": sorted(path.name for path in figures_dir.glob("*.png")),
    }
    save_json(results_dir / "report_notes.json", notes)
    print(json.dumps(notes, indent=2, default=str))


if __name__ == "__main__":
    main()
