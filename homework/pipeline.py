from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class PipelinePaths:
    input_dir: Path
    output_dir: Path
    plots_dir: Path


def _default_paths() -> PipelinePaths:
    repo_root = Path(__file__).resolve().parents[1]
    return PipelinePaths(
        input_dir=repo_root / "files" / "input",
        output_dir=repo_root / "files" / "output",
        plots_dir=repo_root / "files" / "plots",
    )


def build_outputs(
    *,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    plots_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Build required tables and write required artifacts.

    Artifacts produced:
    - files/output/summary.csv
    - files/plots/top10_drivers.png
    """

    defaults = _default_paths()
    paths = PipelinePaths(
        input_dir=Path(input_dir) if input_dir is not None else defaults.input_dir,
        output_dir=Path(output_dir) if output_dir is not None else defaults.output_dir,
        plots_dir=Path(plots_dir) if plots_dir is not None else defaults.plots_dir,
    )

    timesheet = pd.read_csv(paths.input_dir / "timesheet.csv")
    drivers = pd.read_csv(paths.input_dir / "drivers.csv")

    # 1) timesheet_with_means
    mean_hours = (
        timesheet.groupby("driverId", as_index=False)["hours-logged"]
        .mean()
        .rename(columns={"hours-logged": "mean_hours-logged"})
    )
    timesheet_with_means = timesheet.merge(mean_hours, on="driverId", how="left")

    # 2) timesheet_below
    timesheet_below = timesheet_with_means[
        timesheet_with_means["hours-logged"] < timesheet_with_means["mean_hours-logged"]
    ].copy()

    # 3) sum_timesheet
    sum_timesheet = (
        timesheet.groupby("driverId", as_index=False)[["hours-logged", "miles-logged"]]
        .sum()
        .copy()
    )

    # 4) min_max_timesheet
    min_max_timesheet = timesheet.groupby("driverId")["hours-logged"].agg(["min", "max"]).reset_index()
    min_max_timesheet = min_max_timesheet.rename(
        columns={"min": "min_hours-logged", "max": "max_hours-logged"}
    )

    # 5) summary (sum_timesheet + drivers[driverId,name])
    summary = sum_timesheet.merge(drivers[["driverId", "name"]], on="driverId", how="left")
    summary = summary[["driverId", "name", "hours-logged", "miles-logged"]]

    # Write summary.csv
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = paths.output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    # 6) top10 + plot
    top10 = summary.sort_values("miles-logged", ascending=False).head(10).copy()
    top10 = top10.reset_index(drop=True)
    top10["rank"] = top10.index + 1

    def _rank_label(rank: int) -> str:
        # Include a numeric fallback in case emoji glyphs are missing.
        if rank == 1:
            return "🥇 #1"
        if rank == 2:
            return "🥈 #2"
        if rank == 3:
            return "🥉 #3"
        return f"#{rank}"

    top10["label"] = top10["rank"].map(_rank_label) + " " + top10["name"].astype(str)

    # Matplotlib settings (clean look)
    matplotlib.rcParams["font.family"] = "sans-serif"

    paths.plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = paths.plots_dir / "top10_drivers.png"

    # Colors: gold for #1, dark teal for #2, teal gradient for the rest
    gold = "#D4AF37"
    dark_teal = "#0F766E"

    if len(top10) > 2:
        cmap = plt.cm.GnBu
        gradient = [cmap(v) for v in pd.Series(range(len(top10) - 2)).rank(pct=True)]
        colors = [gold, dark_teal, *gradient]
    elif len(top10) == 2:
        colors = [gold, dark_teal]
    elif len(top10) == 1:
        colors = [gold]
    else:
        colors = []

    # Ensure largest on top in a horizontal bar chart: plot reversed order
    plot_df = top10.sort_values("miles-logged", ascending=True)
    plot_colors = [colors[i] for i in plot_df.index] if colors else None

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.set_facecolor("white")

    ax.barh(plot_df["label"], plot_df["miles-logged"], color=plot_colors)

    ax.set_xlabel("Total millas registradas")
    ax.set_ylabel("")
    max_miles = float(plot_df["miles-logged"].max()) if len(plot_df) else 0.0
    left = 130000.0
    if max_miles and max_miles < left:
        left = max_miles * 0.95
    right = max(left + 1.0, max_miles * 1.02)
    ax.set_xlim(left, right)

    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.suptitle("Top 10 Conductores por Millas Registradas", fontsize=14, fontweight="bold")
    ax.set_title("Total de millas acumuladas en 52 semanas", fontsize=10, pad=10)

    plt.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "timesheet": timesheet,
        "drivers": drivers,
        "timesheet_with_means": timesheet_with_means,
        "timesheet_below": timesheet_below,
        "sum_timesheet": sum_timesheet,
        "min_max_timesheet": min_max_timesheet,
        "summary": summary,
        "top10": top10,
    }
