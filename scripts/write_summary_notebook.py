from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path(r"C:\Users\jejsj\PyCharmMiscProject\notebooks\02_experiment_log_summary.ipynb")

NB = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "# Titanic Experiment Log Summary\n\nThis notebook gives a quick, readable overview of the experiment history.\n\nUse it when you want a simple explanation of what improved across runs without using widgets.",
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "from pathlib import Path\n\nimport matplotlib.pyplot as plt\nimport pandas as pd\n\nLOG_PATH = Path(\"../outputs/experiment_log.csv\")\nplt.style.use(\"seaborn-v0_8-whitegrid\")\n",
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "def load_experiment_log(path=LOG_PATH):\n    if not path.exists():\n        return pd.DataFrame()\n\n    log_df = pd.read_csv(path)\n    if log_df.empty:\n        return log_df\n\n    log_df[\"timestamp_utc\"] = pd.to_datetime(log_df[\"timestamp_utc\"], utc=True, errors=\"coerce\")\n    log_df = log_df.dropna(subset=[\"timestamp_utc\"]).sort_values(\"timestamp_utc\").reset_index(drop=True)\n    log_df[\"run_number\"] = log_df.index + 1\n    return log_df\n\n\nlog_df = load_experiment_log()\n\nif log_df.empty:\n    print(\"No experiment log found yet. Run main.py first.\")\nelse:\n    latest = log_df.iloc[-1]\n    summary = pd.DataFrame([\n        {\n            \"Latest run\": int(latest[\"run_number\"]),\n            \"Timestamp UTC\": latest[\"timestamp_utc\"].strftime(\"%Y-%m-%d %H:%M UTC\"),\n            \"Best baseline model\": latest[\"best_baseline_model\"],\n            \"Best engineered model\": latest[\"best_engineered_model\"],\n            \"Top delta model\": latest[\"top_delta_model\"],\n            \"Top F1 delta\": round(float(latest[\"top_f1_delta\"]), 4),\n            \"Mean F1 delta\": round(float(latest[\"mean_f1_delta\"]), 4),\n        }\n    ])\n\n    display(summary)\n",
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "if not log_df.empty:\n    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)\n\n    ax = axes[0]\n    ax.plot(log_df[\"run_number\"], log_df[\"best_baseline_f1\"], marker=\"o\", label=\"Baseline best F1\", color=\"#d95f02\")\n    ax.plot(log_df[\"run_number\"], log_df[\"best_engineered_f1\"], marker=\"o\", label=\"Engineered best F1\", color=\"#1b9e77\")\n    ax.set_title(\"Best model F1 by run\")\n    ax.set_xlabel(\"Run number\")\n    ax.set_ylabel(\"F1-score\")\n    ax.set_ylim(0, 1)\n    ax.grid(True, alpha=0.3)\n    ax.legend(loc=\"lower right\")\n\n    ax = axes[1]\n    ax.bar(log_df[\"run_number\"], log_df[\"top_f1_delta\"], color=\"#4c78a8\")\n    ax.axhline(0, color=\"black\", linewidth=1)\n    ax.set_title(\"Top F1 delta per run\")\n    ax.set_xlabel(\"Run number\")\n    ax.set_ylabel(\"F1 delta\")\n    ax.grid(True, axis=\"y\", alpha=0.3)\n\n    plt.show()\n",
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Key takeaways\n\n- The project starts with a Titanic survival prediction task.\n- Baseline preprocessing becomes stronger with `FamilySize`, `IsAlone`, and `Title`.\n- Logistic Regression often benefits the most from the engineered features.\n- The experiment log makes it easy to compare runs over time.",
        },
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(NB, indent=2), encoding="utf-8")
    print(f"wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()

