from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocess import clean_data
from src.train import get_models, train_model
from src.evaluate import (
    build_subgroup_error_report,
    evaluate,
    extract_feature_importance,
    save_confusion_matrix_plot,
    save_feature_importance_plot,
)
from visualize import save_experiment_history_plot

def build_subgroup_inputs(X_test):
    if "Sex_female" in X_test.columns:
        sex_group = X_test["Sex_female"].map({True: "female", False: "male"}).fillna("unknown")
    else:
        sex_group = pd.Series(["unknown"] * len(X_test), index=X_test.index)

    if "Pclass" in X_test.columns:
        pclass_group = X_test["Pclass"].round().astype(int).map(lambda value: f"Class {value}")
    else:
        pclass_group = pd.Series(["unknown"] * len(X_test), index=X_test.index)

    if "Age" in X_test.columns:
        age_group = pd.cut(
            X_test["Age"],
            bins=[0, 12, 17, 34, 49, float("inf")],
            labels=["0-12", "13-17", "18-34", "35-49", "50+"],
            include_lowest=True,
        ).astype(str)
    else:
        age_group = pd.Series(["unknown"] * len(X_test), index=X_test.index)

    return {
        "sex": sex_group,
        "pclass": pclass_group,
        "age_bin": age_group,
    }


def run_scenario(scenario_name, processed_df, train_idx, test_idx):
    scenario_slug = scenario_name.lower().replace(" ", "_")

    X = processed_df.drop("Survived", axis=1)
    y = processed_df["Survived"]

    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    print(f"=== {scenario_name} ===")
    print(f"Dataset split: {len(X_train)} train samples, {len(X_test)} test samples")
    print(f"Train survival rate: {y_train.mean():.1%}")
    print(f"Test survival rate: {y_test.mean():.1%}")
    print()

    output_dir = Path("outputs/confusion_matrices") / scenario_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_importance_dir = Path("outputs/feature_importance") / scenario_slug
    feature_importance_dir.mkdir(parents=True, exist_ok=True)

    subgroup_dir = Path("outputs/subgroup_errors") / scenario_slug
    subgroup_dir.mkdir(parents=True, exist_ok=True)

    subgroup_inputs = build_subgroup_inputs(X_test)
    results = {}

    for model_name, estimator in get_models().items():
        model = train_model(estimator, X_train, y_train)
        y_test_pred = model.predict(X_test)
        train_metrics = evaluate(model, X_train, y_train)
        test_metrics = evaluate(model, X_test, y_test)
        results[model_name] = {
            "train": train_metrics,
            "test": test_metrics,
        }

        print(f"{model_name} metrics")
        print(f"- Train accuracy : {train_metrics['accuracy']:.4f}")
        print(f"- Test accuracy  : {test_metrics['accuracy']:.4f}")
        print(f"- Train F1-score : {train_metrics['f1']:.4f}")
        print(f"- Test F1-score  : {test_metrics['f1']:.4f}")
        print(f"- Test precision : {test_metrics['precision']:.4f}")
        print(f"- Test recall    : {test_metrics['recall']:.4f}")
        print(f"- Test confusion matrix: {test_metrics['confusion_matrix']}")
        print("- Test classification report:")
        print(test_metrics["classification_report"])
        print(f"- Overfitting gap (accuracy): {train_metrics['accuracy'] - test_metrics['accuracy']:.4f}")
        print(f"- Overfitting gap (F1-score): {train_metrics['f1'] - test_metrics['f1']:.4f}")

        model_slug = model_name.lower().replace(" ", "_")
        confusion_path = output_dir / f"{model_slug}_confusion_matrix.png"
        save_confusion_matrix_plot(
            test_metrics["confusion_matrix"],
            f"{model_name} - {scenario_name} - Test Confusion Matrix",
            confusion_path,
        )
        print(f"- Saved confusion matrix plot: {confusion_path}")

        importance_df = extract_feature_importance(model, X_train.columns)
        if importance_df is not None:
            importance_csv_path = feature_importance_dir / f"{model_slug}_feature_importance.csv"
            importance_plot_path = feature_importance_dir / f"{model_slug}_feature_importance.png"
            importance_df.to_csv(importance_csv_path, index=False)
            save_feature_importance_plot(
                importance_df,
                f"{model_name} - {scenario_name} - Top Feature Importance",
                importance_plot_path,
            )
            print(f"- Saved feature importance CSV : {importance_csv_path}")
            print(f"- Saved feature importance plot: {importance_plot_path}")
        else:
            print("- Feature importance not available for this model.")

        for subgroup_name, subgroup_values in subgroup_inputs.items():
            subgroup_report = build_subgroup_error_report(
                y_test,
                y_test_pred,
                subgroup_values,
                subgroup_name,
            )
            subgroup_file = subgroup_dir / f"{model_slug}_{subgroup_name}_errors.csv"
            subgroup_report.to_csv(subgroup_file, index=False)
            hardest_group = subgroup_report.iloc[0]
            print(
                f"- Highest-error {subgroup_name}: {hardest_group[subgroup_name]} "
                f"(error_rate={hardest_group['error_rate']:.2%}, samples={int(hardest_group['samples'])})"
            )
            print(f"- Saved subgroup error report: {subgroup_file}")

        print()

    best_model_name = max(results, key=lambda name: results[name]["test"]["f1"])
    best_model_test = results[best_model_name]["test"]
    print(f"Best model by test F1-score ({scenario_name}): {best_model_name}")
    print(f"Best model test accuracy: {best_model_test['accuracy']:.4f}")
    print(f"Best model test F1-score: {best_model_test['f1']:.4f}")
    print()

    return results


def get_best_model_summary(results):
    best_model_name = max(results, key=lambda name: results[name]["test"]["f1"])
    best_model_test = results[best_model_name]["test"]
    return {
        "best_model": best_model_name,
        "best_accuracy": best_model_test["accuracy"],
        "best_f1": best_model_test["f1"],
    }


def append_experiment_log(all_results, comparison_df, log_path):
    baseline_summary = get_best_model_summary(all_results["Baseline features"])
    engineered_summary = get_best_model_summary(all_results["Engineered features"])
    best_delta_row = comparison_df.iloc[0]

    log_row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "best_baseline_model": baseline_summary["best_model"],
        "best_baseline_f1": baseline_summary["best_f1"],
        "best_baseline_accuracy": baseline_summary["best_accuracy"],
        "best_engineered_model": engineered_summary["best_model"],
        "best_engineered_f1": engineered_summary["best_f1"],
        "best_engineered_accuracy": engineered_summary["best_accuracy"],
        "top_delta_model": best_delta_row["model"],
        "top_accuracy_delta": best_delta_row["accuracy_delta"],
        "top_precision_delta": best_delta_row["precision_delta"],
        "top_recall_delta": best_delta_row["recall_delta"],
        "top_f1_delta": best_delta_row["f1_delta"],
        "mean_accuracy_delta": comparison_df["accuracy_delta"].mean(),
        "mean_f1_delta": comparison_df["f1_delta"].mean(),
    }

    log_df = pd.DataFrame([log_row])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)


raw_df = pd.read_csv("data/raw/train.csv")

train_idx, test_idx = train_test_split(
    raw_df.index,
    test_size=0.2,
    random_state=42,
    stratify=raw_df["Survived"],
)

scenario_frames = {
    "Baseline features": clean_data(raw_df, use_feature_engineering=False),
    "Engineered features": clean_data(raw_df, use_feature_engineering=True),
}

all_results = {}
for scenario_name, processed_df in scenario_frames.items():
    all_results[scenario_name] = run_scenario(scenario_name, processed_df, train_idx, test_idx)

comparison_rows = []
for model_name in get_models().keys():
    baseline_test = all_results["Baseline features"][model_name]["test"]
    engineered_test = all_results["Engineered features"][model_name]["test"]
    comparison_rows.append(
        {
            "model": model_name,
            "baseline_accuracy": baseline_test["accuracy"],
            "engineered_accuracy": engineered_test["accuracy"],
            "accuracy_delta": engineered_test["accuracy"] - baseline_test["accuracy"],
            "baseline_precision": baseline_test["precision"],
            "engineered_precision": engineered_test["precision"],
            "precision_delta": engineered_test["precision"] - baseline_test["precision"],
            "baseline_recall": baseline_test["recall"],
            "engineered_recall": engineered_test["recall"],
            "recall_delta": engineered_test["recall"] - baseline_test["recall"],
            "baseline_f1": baseline_test["f1"],
            "engineered_f1": engineered_test["f1"],
            "f1_delta": engineered_test["f1"] - baseline_test["f1"],
        }
    )

comparison_df = pd.DataFrame(comparison_rows).sort_values("f1_delta", ascending=False)
comparison_path = Path("outputs/feature_engineering_comparison.csv")
comparison_df.to_csv(comparison_path, index=False)

experiment_log_path = Path("outputs/experiment_log.csv")
append_experiment_log(all_results, comparison_df, experiment_log_path)

history_path = Path("outputs/experiment_history.png")
if save_experiment_history_plot(experiment_log_path, history_path):
    print(f"Saved experiment history plot: {history_path}")
else:
    print("- Experiment history plot not available yet.")

print("=== Feature Engineering Impact (Engineered - Baseline) ===")
print(
    comparison_df[
        [
            "model",
            "accuracy_delta",
            "precision_delta",
            "recall_delta",
            "f1_delta",
        ]
    ].to_string(index=False)
)
print(f"Saved comparison CSV: {comparison_path}")
print(f"Appended experiment log: {experiment_log_path}")
print("Training completed successfully!")
