from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path(r"C:\Users\jejsj\PyCharmMiscProject\notebooks\03_experiment_log_dashboard.ipynb")


def build_notebook() -> dict:
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "# Titanic Experiment Log Dashboard\n\nThis notebook helps you read the project quickly and explore the training log interactively.\n\nIt focuses on `outputs/experiment_log.csv`, which is appended every time `main.py` runs. Use the widgets below to filter runs and compare the baseline vs engineered feature scenarios.",
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": (
                    "from pathlib import Path\n\n"
                    "import matplotlib.pyplot as plt\n"
                    "import pandas as pd\n"
                    "import ipywidgets as widgets\n"
                    "from IPython.display import Markdown, clear_output, display\n\n"
                    "LOG_PATH = Path(\"../outputs/experiment_log.csv\")\n"
                    "plt.style.use(\"seaborn-v0_8-whitegrid\")\n"
                ),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": (
                    "def load_experiment_log(path=LOG_PATH):\n"
                    "    if not path.exists():\n"
                    "        return pd.DataFrame()\n\n"
                    "    log_df = pd.read_csv(path)\n"
                    "    if log_df.empty:\n"
                    "        return log_df\n\n"
                    "    log_df[\"timestamp_utc\"] = pd.to_datetime(log_df[\"timestamp_utc\"], utc=True, errors=\"coerce\")\n"
                    "    log_df = log_df.dropna(subset=[\"timestamp_utc\"]).sort_values(\"timestamp_utc\").reset_index(drop=True)\n"
                    "    log_df[\"run_number\"] = log_df.index + 1\n"
                    "    return log_df\n\n\n"
                    "log_df = load_experiment_log()\n\n"
                    "project_map = pd.DataFrame([\n"
                    "    {\"Step\": \"Load data\", \"File\": \"main.py\", \"What it does\": \"Reads Titanic CSV and creates train/test splits\"},\n"
                    "    {\"Step\": \"Clean data\", \"File\": \"src/preprocess.py\", \"What it does\": \"Fills missing values and creates engineered features\"},\n"
                    "    {\"Step\": \"Train models\", \"File\": \"src/train.py\", \"What it does\": \"Fits LogisticRegression, RandomForest, GradientBoosting\"},\n"
                    "    {\"Step\": \"Evaluate\", \"File\": \"src/evaluate.py\", \"What it does\": \"Computes accuracy, precision, recall, F1, reports, and plots\"},\n"
                    "    {\"Step\": \"Experiment log\", \"File\": \"outputs/experiment_log.csv\", \"What it does\": \"Tracks best baseline vs engineered results over time\"},\n"
                    "])\n\n"
                    "if log_df.empty:\n"
                    "    display(Markdown(\"### No experiment log yet\\nRun `python main.py` first, then return to this notebook.\"))\n"
                    "else:\n"
                    "    latest_summary = log_df.iloc[[-1]][[\n"
                    "        \"run_number\",\n"
                    "        \"timestamp_utc\",\n"
                    "        \"best_baseline_model\",\n"
                    "        \"best_engineered_model\",\n"
                    "        \"top_delta_model\",\n"
                    "        \"top_f1_delta\",\n"
                    "        \"mean_f1_delta\",\n"
                    "    ]].copy()\n"
                    "    latest_summary[\"timestamp_utc\"] = latest_summary[\"timestamp_utc\"].dt.strftime(\"%Y-%m-%d %H:%M UTC\")\n\n"
                    "    display(Markdown(f\"### Loaded {len(log_df)} experiment runs\"))\n"
                    "    display(latest_summary)\n\n"
                    "display(Markdown(\"### Project map\"))\n"
                    "display(project_map)\n"
                ),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": (
                    "if not log_df.empty:\n"
                    "    metric_options = [\n"
                    "        (\"Top F1 delta\", \"top_f1_delta\"),\n"
                    "        (\"Mean F1 delta\", \"mean_f1_delta\"),\n"
                    "        (\"Top accuracy delta\", \"top_accuracy_delta\"),\n"
                    "        (\"Mean accuracy delta\", \"mean_accuracy_delta\"),\n"
                    "        (\"Best baseline F1\", \"best_baseline_f1\"),\n"
                    "        (\"Best engineered F1\", \"best_engineered_f1\"),\n"
                    "    ]\n\n"
                    "    metric_dropdown = widgets.Dropdown(\n"
                    "        options=metric_options,\n"
                    "        value=\"top_f1_delta\",\n"
                    "        description=\"Metric:\",\n"
                    "        layout=widgets.Layout(width=\"320px\"),\n"
                    "    )\n\n"
                    "    run_slider = widgets.IntRangeSlider(\n"
                    "        value=(1, int(log_df[\"run_number\"].max())),\n"
                    "        min=1,\n"
                    "        max=int(log_df[\"run_number\"].max()),\n"
                    "        step=1,\n"
                    "        description=\"Runs:\",\n"
                    "        continuous_update=False,\n"
                    "        layout=widgets.Layout(width=\"520px\"),\n"
                    "    )\n\n"
                    "    output = widgets.Output()\n\n"
                    "    def render_dashboard(metric, run_window):\n"
                    "        start_run, end_run = run_window\n"
                    "        filtered = log_df[(log_df[\"run_number\"] >= start_run) & (log_df[\"run_number\"] <= end_run)].copy()\n\n"
                    "        with output:\n"
                    "            clear_output(wait=True)\n\n"
                    "            if filtered.empty:\n"
                    "                display(Markdown(\"No experiment rows are available in the selected run range.\"))\n"
                    "                return\n\n"
                    "            fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)\n"
                    "            fig.suptitle(\"Titanic Experiment History\", fontsize=18, fontweight=\"bold\")\n\n"
                    "            ax = axes[0]\n"
                    "            ax.plot(filtered[\"run_number\"], filtered[\"best_baseline_f1\"], marker=\"o\", label=\"Baseline best F1\", color=\"#d95f02\")\n"
                    "            ax.plot(filtered[\"run_number\"], filtered[\"best_engineered_f1\"], marker=\"o\", label=\"Engineered best F1\", color=\"#1b9e77\")\n"
                    "            ax.set_title(\"Best model F1 by run\")\n"
                    "            ax.set_ylabel(\"F1-score\")\n"
                    "            ax.set_ylim(0, 1)\n"
                    "            ax.grid(True, alpha=0.3)\n"
                    "            ax.legend(loc=\"lower right\")\n\n"
                    "            ax = axes[1]\n"
                    "            ax.bar(filtered[\"run_number\"], filtered[metric], color=\"#4c78a8\")\n"
                    "            ax.axhline(0, color=\"black\", linewidth=1)\n"
                    "            ax.set_title(f\"{metric} across runs\")\n"
                    "            ax.set_xlabel(\"Run number\")\n"
                    "            ax.set_ylabel(metric)\n"
                    "            ax.grid(True, axis=\"y\", alpha=0.3)\n\n"
                    "            plt.show()\n\n"
                    "            preview_cols = [\n"
                    "                \"run_number\",\n"
                    "                \"timestamp_utc\",\n"
                    "                \"best_baseline_model\",\n"
                    "                \"best_engineered_model\",\n"
                    "                \"top_delta_model\",\n"
                    "                metric,\n"
                    "            ]\n"
                    "            preview = filtered[preview_cols].tail(8).copy()\n"
                    "            preview[\"timestamp_utc\"] = preview[\"timestamp_utc\"].dt.strftime(\"%Y-%m-%d %H:%M UTC\")\n"
                    "            display(preview)\n\n"
                    "    controls = widgets.HBox([metric_dropdown, run_slider])\n"
                    "    display(Markdown(\"### Interactive experiment dashboard\"))\n"
                    "    display(controls, output)\n\n"
                    "    def _update(change=None):\n"
                    "        render_dashboard(metric_dropdown.value, run_slider.value)\n\n"
                    "    metric_dropdown.observe(_update, names=\"value\")\n"
                    "    run_slider.observe(_update, names=\"value\")\n"
                    "    _update()\n"
                    "else:\n"
                    "    display(Markdown(\"### Interactive experiment dashboard\\nRun `python main.py` first so this dashboard has a log to display.\"))\n"
                ),
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "## How to read this project\n\n- `main.py` is the main training entrypoint. It compares baseline and engineered features.\n- `src/preprocess.py` prepares the Titanic data and adds `FamilySize`, `IsAlone`, and `Title`.\n- `src/train.py` trains three models: Logistic Regression, Random Forest, and Gradient Boosting.\n- `src/evaluate.py` prints metrics, confusion matrices, feature importance, and subgroup error reports.\n- `outputs/experiment_log.csv` keeps a run history. This notebook turns that history into an interactive dashboard.\n\nTip: use the **Runs** slider to focus on a smaller window and the **Metric** dropdown to change the chart on the lower panel.",
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
    NOTEBOOK_PATH.write_text(json.dumps(build_notebook(), indent=2), encoding="utf-8")
    print(f"wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()

