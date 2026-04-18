from __future__ import annotations

from pathlib import Path


def pytest_sessionstart(session):  # type: ignore[no-untyped-def]
    repo_root = Path(__file__).resolve().parent
    summary_path = repo_root / "files" / "output" / "summary.csv"
    plot_path = repo_root / "files" / "plots" / "top10_drivers.png"

    if summary_path.exists() and plot_path.exists():
        return

    from homework.pipeline import build_outputs

    build_outputs()
