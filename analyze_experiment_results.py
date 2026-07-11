"""Plot experiment histories and build grouped summary DataFrames.

The four groups are:
    1. resnet: names beginning with ``resnet_``
    2. base: names beginning with ``base_``
    3. build: every remaining name
    4. base_and_build: the concatenation of groups 2 and 3

Any experiment whose name contains ResNet-18 (including resnet18, resnet_18,
or resnet-18) is excluded from every group.

History CSV files contain validation curves, while each summary JSON contains
one final test result. Therefore, every history figure uses solid validation
curves and matching dashed horizontal lines for final test metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable

# Matplotlib needs a writable cache directory in some notebook/server setups.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ipcv-matplotlib-cache")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "ipcv-xdg-cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY_DIR = ROOT / "outputs" / "histories"
DEFAULT_SUMMARY_DIR = ROOT / "outputs" / "summaries"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "analysis"

GROUP_ORDER = ("resnet", "base", "build", "base_and_build")
GROUP_TITLES = {
    "resnet": "ResNet experiments",
    "base": "Base experiments",
    "build": "Building model experiments",
    "base_and_build": "Base + building notebook experiments",
}

REQUIRED_HISTORY_COLUMNS = {"epoch", "val_loss", "val_acc"}
RESNET18_PATTERN = re.compile(r"resnet[\s_-]*18", flags=re.IGNORECASE)
MAIN_RUN_PATTERN = re.compile(r"resnet[\s_-]*18", flags=re.IGNORECASE)

DUPLICATE_COPY_PATTERN = re.compile(r"\s*\(\d+\)$")


def is_resnet18(name: str) -> bool:
    """Return True for names containing resnet18/resnet_18/resnet-18."""

    return bool(RESNET18_PATTERN.search(str(name)))


def classify_experiment(name: str) -> str | None:
    """Classify an experiment name, or return None when it must be excluded."""

    normalized = str(name).strip().lower()
    if is_resnet18(normalized):
        return None
    if normalized.startswith("resnet_"):
        return "resnet"
    if normalized.startswith("base_"):
        return "base"
    if normalized.startswith("build_"):
        return "build"
    return "build"


def _canonical_history_files(history_dir: Path) -> list[Path]:
    """Return CSV files, ignoring numbered copies when a canonical file exists."""

    csv_files = sorted(history_dir.glob("*.csv"))
    stems = {path.stem for path in csv_files}
    selected: list[Path] = []

    for path in csv_files:
        canonical_stem = DUPLICATE_COPY_PATTERN.sub("", path.stem)
        if canonical_stem != path.stem and canonical_stem in stems:
            continue
        selected.append(path)

    return selected


def load_histories(history_dir: Path) -> dict[str, pd.DataFrame]:
    """Read every canonical history CSV and return it by experiment name."""

    histories: dict[str, pd.DataFrame] = {}
    for path in _canonical_history_files(history_dir):
        experiment = path.stem
        if is_resnet18(experiment):
            continue

        history = pd.read_csv(path)
        missing = REQUIRED_HISTORY_COLUMNS.difference(history.columns)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"{path} is missing required columns: {missing_text}")

        history = history.copy()
        history["experiment"] = experiment
        history["source_file"] = str(path)
        histories[experiment] = history

    return histories


def load_all_summaries(summary_dir: Path) -> pd.DataFrame:
    """Read and flatten all summary JSON files into one pandas DataFrame."""

    records: list[dict] = []
    for path in sorted(summary_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as file:
            record = json.load(file)

        if not isinstance(record, dict):
            raise ValueError(f"Expected a JSON object in {path}")

        record = dict(record)
        experiment = path.stem.removesuffix("_summary")
        record.setdefault("run_name", experiment)
        # The filename is canonical for grouping because some JSON run_name
        # values still contain an older name (for example resnet18_backbone_*).
        record["experiment"] = experiment
        record["source_file"] = str(path)
        records.append(record)

    if not records:
        return pd.DataFrame()

    # cfg fields become columns such as cfg.lr, cfg.batch_size, and cfg.dropout.
    return pd.json_normalize(records, sep=".")


def make_summary_dataframes(all_summaries: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return the four requested grouped summary DataFrames."""

    if all_summaries.empty:
        return {group: all_summaries.copy() for group in GROUP_ORDER}
    if "experiment" not in all_summaries.columns:
        raise ValueError("The summary data does not contain an experiment column")

    summaries = all_summaries.copy()
    summaries["group"] = summaries["experiment"].map(classify_experiment)
    summaries = summaries.loc[summaries["group"].notna()].copy()

    grouped = {
        "resnet": summaries.loc[summaries["group"].eq("resnet")].copy(),
        "base": summaries.loc[summaries["group"].eq("base")].copy(),
        "build": summaries.loc[summaries["group"].eq("build")].copy(),
    }
    grouped["base_and_build"] = pd.concat(
        [grouped["base"], grouped["build"]], ignore_index=True
    )

    for group, frame in grouped.items():
        grouped[group] = frame.sort_values("experiment").reset_index(drop=True)

    return grouped


def make_history_groups(
    histories: dict[str, pd.DataFrame],
) -> dict[str, dict[str, pd.DataFrame]]:
    """Return the four requested groups of history DataFrames."""

    grouped: dict[str, dict[str, pd.DataFrame]] = {
        "resnet": {},
        "base": {},
        "build": {},
    }
    for name, history in histories.items():
        group = classify_experiment(name)
        if group is not None:
            grouped[group][name] = history

    grouped["base_and_build"] = {
        **grouped["base"],
        **grouped["build"],
    }
    return grouped


def _normalized_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _summary_lookup(all_summaries: pd.DataFrame) -> dict[str, pd.Series]:
    """Index summaries using canonical filenames and legacy run_name aliases."""

    if all_summaries.empty or "experiment" not in all_summaries.columns:
        return {}

    lookup: dict[str, pd.Series] = {}
    included_rows: list[pd.Series] = []
    for _, row in all_summaries.iterrows():
        experiment = str(row["experiment"])
        if is_resnet18(experiment):
            continue
        included_rows.append(row)
        lookup[experiment] = row
        lookup.setdefault(str(row.get("run_name", experiment)), row)

    # Add separator-insensitive aliases and unique prefix aliases. The latter
    # pairs "basic cnn" with the legacy summary name "basic_cnn_residual".
    normalized_rows: dict[str, list[pd.Series]] = {}
    for row in included_rows:
        for alias in (row["experiment"], row.get("run_name", row["experiment"])):
            normalized_rows.setdefault(_normalized_name(str(alias)), []).append(row)

    for normalized, rows in normalized_rows.items():
        unique_rows = {str(row["source_file"]): row for row in rows}
        if len(unique_rows) == 1:
            lookup.setdefault(normalized, next(iter(unique_rows.values())))

    return lookup


def _find_summary(name: str, lookup: dict[str, pd.Series]) -> pd.Series | None:
    if name in lookup:
        return lookup[name]

    normalized = _normalized_name(name)
    if normalized in lookup:
        return lookup[normalized]

    prefix_matches: dict[str, pd.Series] = {}
    if len(normalized) >= 5:
        for alias, row in lookup.items():
            normalized_alias = _normalized_name(alias)
            if normalized_alias.startswith(normalized) or normalized.startswith(
                normalized_alias
            ):
                prefix_matches[str(row["source_file"])] = row
    if len(prefix_matches) == 1:
        return next(iter(prefix_matches.values()))
    return None


def plot_history_group(
    group_name: str,
    histories: dict[str, pd.DataFrame],
    all_summaries: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot validation histories and final test metrics for one group."""

    if not histories:
        print(f"Skipping empty history group: {group_name}")
        return

    summary_by_name = _summary_lookup(all_summaries)
    names = sorted(histories)
    colors = plt.colormaps["turbo"].resampled(max(len(names), 2))

    longest_name = max(map(len, names))
    figure_width = max(16.0, min(24.0, 13.0 + 0.11 * longest_name))
    figure_height = max(8.0, 3.2 + 0.18 * len(names))
    legend_fraction = min(0.48, max(0.26, 0.13 + 0.006 * longest_name))
    plot_right = 1.0 - legend_fraction

    fig, axes = plt.subplots(
        2, 1, figsize=(figure_width, figure_height), sharex=True
    )
    accuracy_axis, loss_axis = axes

    for index, name in enumerate(names):
        history = histories[name]
        color = colors(index)
        accuracy_axis.plot(
            history["epoch"],
            history["val_acc"],
            color=color,
            linewidth=1.8,
            label=name,
        )
        loss_axis.plot(
            history["epoch"],
            history["val_loss"],
            color=color,
            linewidth=1.8,
            label=name,
        )

        summary = _find_summary(name, summary_by_name)
        if summary is not None:
            test_acc = pd.to_numeric(summary.get("test_acc"), errors="coerce")
            test_loss = pd.to_numeric(summary.get("test_loss"), errors="coerce")
            if pd.notna(test_acc):
                accuracy_axis.axhline(
                    test_acc, color=color, linestyle="--", linewidth=1.1, alpha=0.55
                )
            if pd.notna(test_loss):
                loss_axis.axhline(
                    test_loss, color=color, linestyle="--", linewidth=1.1, alpha=0.55
                )

    title = GROUP_TITLES[group_name]
    fig.suptitle(
        f"{title}\nSolid = validation history; dashed = final test metric",
        fontsize=14,
        fontweight="bold",
    )
    accuracy_axis.set_ylabel("Accuracy")
    loss_axis.set_ylabel("Loss")
    loss_axis.set_xlabel("Epoch")

    for axis in axes:
        axis.grid(True, linestyle=":", alpha=0.45)

    handles, labels = accuracy_axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(plot_right + 0.01, 0.5),
        fontsize=7.5 if len(names) > 25 else 8.0,
        frameon=False,
    )

    fig.tight_layout(rect=(0.01, 0.01, plot_right, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def export_summary_dataframes(
    summary_dataframes: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    """Save the four DataFrames as CSV files for convenient reuse."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for group in GROUP_ORDER:
        summary_dataframes[group].to_csv(
            output_dir / f"summary_{group}.csv", index=False
        )


def run_analysis(
    history_dir: Path = DEFAULT_HISTORY_DIR,
    summary_dir: Path = DEFAULT_SUMMARY_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, pd.DataFrame]:
    """Run the complete analysis and return the four summary DataFrames."""

    if not history_dir.is_dir():
        raise FileNotFoundError(f"History directory does not exist: {history_dir}")
    if not summary_dir.is_dir():
        raise FileNotFoundError(f"Summary directory does not exist: {summary_dir}")

    histories = load_histories(history_dir)
    all_summaries = load_all_summaries(summary_dir)
    history_groups = make_history_groups(histories)
    summary_dataframes = make_summary_dataframes(all_summaries)

    output_dir.mkdir(parents=True, exist_ok=True)
    for group in GROUP_ORDER:
        plot_history_group(
            group,
            history_groups[group],
            all_summaries,
            output_dir / f"history_{group}.png",
        )

    export_summary_dataframes(summary_dataframes, output_dir)

    print(f"Read {len(histories)} non-ResNet-18 history files.")
    print(f"Read all {len(all_summaries)} summary JSON files.")
    for group in GROUP_ORDER:
        print(
            f"{group:>14}: "
            f"{len(history_groups[group]):>2} histories, "
            f"{len(summary_dataframes[group]):>2} summary rows"
        )
    print(f"Outputs written to: {output_dir}")

    return summary_dataframes


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()

    # These are the four requested pandas DataFrames. In a notebook, call
    # `summary_dfs = run_analysis()` and access them with the same keys.
    summary_dfs = run_analysis(args.history_dir, args.summary_dir, args.output_dir)
    resnet_summary_df = summary_dfs["resnet"]
    base_summary_df = summary_dfs["base"]
    build_summary_df = summary_dfs["build"]
    base_and_build_summary_df = summary_dfs["base_and_build"]

    # Keep explicit variable names for notebook/debugger inspection.
    _ = (
        resnet_summary_df,
        base_summary_df,
        build_summary_df,
        base_and_build_summary_df,
    )


if __name__ == "__main__":
    main()
