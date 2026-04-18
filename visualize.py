from pathlib import Path
from typing import cast

import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = Path("data/raw/train.csv")
OUTPUT_PATH = Path("outputs/titanic_overview.png")
EXPERIMENT_LOG_PATH = Path("outputs/experiment_log.csv")
EXPERIMENT_HISTORY_PATH = Path("outputs/experiment_history.png")


def load_data() -> pd.DataFrame:
    return cast(pd.DataFrame, pd.read_csv(DATA_PATH))  # type: ignore[call-overload]


def annotate_bars(ax) -> None:
    for bar in ax.patches:
        height = bar.get_height()
        if pd.notna(height):
            ax.annotate(
                f"{height:.0f}",
                (bar.get_x() + bar.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 3),
                textcoords="offset points",
            )


def build_dashboard(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle("Titanic Data Overview", fontsize=18, fontweight="bold")

    # 1) Survival counts
    survival_labels = df["Survived"].map({0: "Did not survive", 1: "Survived"})
    survival_counts = survival_labels.value_counts().reindex(["Did not survive", "Survived"])
    ax = axes[0, 0]
    survival_counts.plot(kind="bar", ax=ax, color=["#d95f02", "#1b9e77"])
    ax.set_title("Survival Count")
    ax.set_xlabel("")
    ax.set_ylabel("Passengers")
    ax.tick_params(axis="x", rotation=0)
    annotate_bars(ax)

    # 2) Age distribution
    ax = axes[0, 1]
    ages = df["Age"].dropna()
    ax.hist(ages, bins=20, color="#4c78a8", edgecolor="white")
    ax.set_title("Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Passengers")

    # 3) Fare by class
    ax = axes[1, 0]
    class_data = [df.loc[df["Pclass"] == pclass, "Fare"].dropna() for pclass in [1, 2, 3]]
    ax.boxplot(class_data, tick_labels=["1st", "2nd", "3rd"], showfliers=False)
    ax.set_title("Fare by Passenger Class")
    ax.set_xlabel("Passenger Class")
    ax.set_ylabel("Fare")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)

    # 4) Survival rate by sex and class
    ax = axes[1, 1]
    survival_rate = df.pivot_table(index="Sex", columns="Pclass", values="Survived", aggfunc="mean")
    survival_rate = survival_rate.loc[["female", "male"]]
    survival_rate.columns = ["1st", "2nd", "3rd"]
    survival_rate.plot(kind="bar", ax=ax, color=["#66c2a5", "#fc8d62", "#8da0cb"])
    ax.set_title("Survival Rate by Sex and Class")
    ax.set_xlabel("")
    ax.set_ylabel("Survival rate")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Class", loc="upper right")

    return fig


def save_experiment_history_plot(
    log_path: Path = EXPERIMENT_LOG_PATH,
    output_path: Path = EXPERIMENT_HISTORY_PATH,
) -> bool:
    if not log_path.exists():
        return False

    log_df = pd.read_csv(log_path)
    if log_df.empty:
        return False

    log_df["timestamp_utc"] = pd.to_datetime(log_df["timestamp_utc"], utc=True, errors="coerce")
    log_df = log_df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    if log_df.empty:
        return False

    log_df = log_df.reset_index(drop=True)
    log_df["run_number"] = log_df.index + 1

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    fig.suptitle("Experiment History", fontsize=18, fontweight="bold")

    ax = axes[0]
    ax.plot(log_df["run_number"], log_df["best_baseline_f1"], marker="o", label="Baseline best F1")
    ax.plot(log_df["run_number"], log_df["best_engineered_f1"], marker="o", label="Engineered best F1")
    ax.set_ylabel("Best F1-score")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.bar(log_df["run_number"], log_df["top_f1_delta"], color="#4c78a8")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Run number")
    ax.set_ylabel("Top F1 delta")
    ax.grid(True, axis="y", alpha=0.3)

    ax.set_xticks(log_df["run_number"])
    ax.set_xticklabels([
        ts.strftime("%m-%d %H:%M") for ts in log_df["timestamp_utc"].dt.tz_convert(None)
    ], rotation=45, ha="right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    df = load_data()
    fig = build_dashboard(df)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

