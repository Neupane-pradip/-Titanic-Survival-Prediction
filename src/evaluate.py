import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate(model, X_data, y_data):
    preds = model.predict(X_data)
    cm = confusion_matrix(y_data, preds)
    return {
        "accuracy": accuracy_score(y_data, preds),
        "precision": precision_score(y_data, preds, pos_label=1, zero_division=0),
        "recall": recall_score(y_data, preds, pos_label=1, zero_division=0),
        "f1": f1_score(y_data, preds, pos_label=1, zero_division=0),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_data, preds, zero_division=0),
    }


def save_confusion_matrix_plot(confusion_values, title, output_path):
    fig, ax = plt.subplots(figsize=(4, 4))
    image = ax.imshow(confusion_values, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticklabels(["No", "Yes"])

    for i, row in enumerate(confusion_values):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center", color="black")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def extract_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coefs = model.coef_
        if getattr(coefs, "ndim", 1) > 1:
            coefs = coefs[0]
        importances = abs(coefs)
    else:
        return None

    importance_df = pd.DataFrame(
        {
            "feature": list(feature_names),
            "importance": importances,
        }
    )
    return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)


def save_feature_importance_plot(importance_df, title, output_path, top_n=10):
    plot_df = importance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(plot_df["feature"], plot_df["importance"], color="#4c78a8")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_subgroup_error_report(y_true, y_pred, subgroup_series, subgroup_name):
    report_df = pd.DataFrame(
        {
            subgroup_name: subgroup_series,
            "actual": y_true,
            "predicted": y_pred,
        }
    )
    report_df["is_error"] = (report_df["actual"] != report_df["predicted"]).astype(int)

    summary = (
        report_df.groupby(subgroup_name, dropna=False)
        .agg(
            samples=("is_error", "size"),
            errors=("is_error", "sum"),
            error_rate=("is_error", "mean"),
            actual_survival_rate=("actual", "mean"),
            predicted_survival_rate=("predicted", "mean"),
        )
        .sort_values("error_rate", ascending=False)
        .reset_index()
    )

    return summary


