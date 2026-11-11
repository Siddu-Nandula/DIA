
"""Generate charts from tabular documents."""

import os
import threading
from functools import wraps
from pathlib import Path
from tempfile import NamedTemporaryFile

import matplotlib

matplotlib.use("Agg")  # Allows chart generation without a GUI.

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from matplotlib.ticker import FuncFormatter

from app.data_analytics import get_dataframe
from app.document_loader import DocumentResult


# Use an absolute project-level output directory.
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

# Matplotlib is not thread-safe. Protect chart generation.
_CHART_LOCK = threading.RLock()


def _synchronized(func):
    """Prevent simultaneous chart generation from different threads."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with _CHART_LOCK:
            return func(*args, **kwargs)

    return wrapper


def _validate_filename(filename: str) -> None:
    """Validate chart filenames."""

    if (
        not filename
        or Path(filename).name != filename
        or not filename.lower().endswith(".png")
    ):
        raise ValueError("Filename must be a simple .png filename.")


def _save_figure_atomic(fig, filename: str) -> str:
    """Save a chart completely before replacing the final output."""

    _validate_filename(filename)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / filename
    temp_path = None

    try:
        # Create the temporary image in the same directory.
        with NamedTemporaryFile(
            suffix=".png",
            prefix="chart_",
            dir=OUTPUT_DIR,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

        # Save the complete chart to the temporary file.
        fig.savefig(
            temp_path,
            format="png",
            dpi=150,
            bbox_inches="tight",
        )

        # Atomically replace the final chart.
        os.replace(temp_path, output_path)

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    return str(output_path)


@_synchronized
def generate_bar_chart(
    document: DocumentResult,
    category: str,
    metric: str,
    operation: str = "sum",
    filename: str = "bar_chart.png",
    sheet_name: str | None = None,
) -> str:
    """Generate an aggregated bar chart from a selected worksheet."""

    df = get_dataframe(document, sheet_name=sheet_name)

    if category not in df.columns:
        raise ValueError(f"Category column '{category}' does not exist.")

    if metric not in df.columns:
        raise ValueError(f"Metric column '{metric}' does not exist.")

    if not pd.api.types.is_numeric_dtype(df[metric]):
        raise ValueError(f"Metric column '{metric}' must be numeric.")

    if operation not in {"sum", "mean", "count"}:
        raise ValueError("Operation must be sum, mean, or count.")

    _validate_filename(filename)

    grouped = df.groupby(category, dropna=False)[metric].agg(operation)
    grouped = grouped.sort_values(ascending=False)

    if grouped.empty:
        raise ValueError("No data available to plot.")

    fig, ax = plt.subplots(figsize=(9, 5))

    try:
        grouped.plot(kind="bar", ax=ax)

        worksheet_label = f" — {sheet_name}" if sheet_name else ""

        ax.set_title(
            f"{operation.title()} of {metric} by {category}{worksheet_label}"
        )
        ax.set_xlabel(category.title())
        ax.set_ylabel(f"{operation.title()} {metric}")
        ax.tick_params(axis="x", rotation=30)

        fig.tight_layout()

        return _save_figure_atomic(fig, filename)

    finally:
        plt.close(fig)


@_synchronized
def generate_histogram(
    document: DocumentResult,
    metric: str,
    filename: str = "histogram.png",
    sheet_name: str | None = None,
) -> str:
    """Generate a histogram from a selected worksheet's numeric column."""

    df = get_dataframe(document, sheet_name=sheet_name)

    if metric not in df.columns:
        raise ValueError(f"Column '{metric}' does not exist.")

    if not pd.api.types.is_numeric_dtype(df[metric]):
        raise ValueError(f"Column '{metric}' must be numeric.")

    _validate_filename(filename)

    values = df[metric].dropna()

    if values.empty:
        raise ValueError(f"Column '{metric}' has no numeric values.")

    fig, ax = plt.subplots(figsize=(9, 5))

    try:
        worksheet_label = f" — {sheet_name}" if sheet_name else ""

        ax.hist(values, bins="auto", edgecolor="black")
        ax.set_title(f"Distribution of {metric}{worksheet_label}")
        ax.set_xlabel(metric.title())
        ax.set_ylabel("Frequency")

        fig.tight_layout()

        return _save_figure_atomic(fig, filename)

    finally:
        plt.close(fig)


@_synchronized
def generate_line_chart(
    document: DocumentResult,
    x_axis: str,
    metric: str,
    filename: str = "line_chart.png",
    sheet_name: str | None = None,
    frequency: str | None = None,
    operation: str = "sum",
) -> str:
    """Generate a line chart, optionally aggregating values over time."""

    df = get_dataframe(document, sheet_name=sheet_name)

    if x_axis not in df.columns:
        raise ValueError(f"X-axis column '{x_axis}' does not exist.")

    if metric not in df.columns:
        raise ValueError(f"Metric column '{metric}' does not exist.")

    if not pd.api.types.is_numeric_dtype(df[metric]):
        raise ValueError(f"Metric column '{metric}' must be numeric.")

    _validate_filename(filename)

    if frequency is not None:
        if frequency not in {"MS", "W", "D", "YS"}:
            raise ValueError("Frequency must be MS, W, D, or YS.")

        if operation not in {"sum", "mean", "count"}:
            raise ValueError("Operation must be sum, mean, or count.")

    plot_df = df[[x_axis, metric]].dropna().copy()

    if plot_df.empty:
        raise ValueError("No data available to plot.")

    if frequency is not None:
        plot_df[x_axis] = pd.to_datetime(
            plot_df[x_axis],
            errors="coerce",
        )

        plot_df = plot_df.dropna(subset=[x_axis])

        if plot_df.empty:
            raise ValueError("No valid dates available to plot.")

        grouped = plot_df.groupby(
            pd.Grouper(key=x_axis, freq=frequency)
        )

        aggregated = grouped[metric].agg(operation)

        observed_periods = grouped.size()
        observed_periods = observed_periods[observed_periods > 0]

        plot_df = (
            aggregated.loc[observed_periods.index]
            .rename(metric)
            .reset_index()
        )

    elif pd.api.types.is_datetime64_any_dtype(plot_df[x_axis]):
        plot_df = plot_df.sort_values(by=x_axis)

    if plot_df.empty:
        raise ValueError("No data available to plot.")

    fig, ax = plt.subplots(figsize=(12, 6))

    try:
        ax.plot(
            plot_df[x_axis],
            plot_df[metric],
            marker="o",
            linewidth=2,
        )

        worksheet_label = f" — {sheet_name}" if sheet_name else ""

        if frequency is not None:
            frequency_label = {
                "MS": "Monthly",
                "W": "Weekly",
                "D": "Daily",
                "YS": "Yearly",
            }[frequency]

            title = (
                f"{frequency_label} {operation.title()} "
                f"of {metric.title()}"
            )
        else:
            title = f"{metric.title()} over {x_axis.title()}"

        ax.set_title(f"{title}{worksheet_label}")
        ax.set_xlabel(x_axis.title())
        ax.set_ylabel(metric.title())

        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, position: f"{value:,.0f}")
        )

        if frequency == "MS":
            ax.xaxis.set_major_locator(
                mdates.MonthLocator(interval=1)
            )
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter("%b %Y")
            )

        elif frequency == "YS":
            ax.xaxis.set_major_locator(
                mdates.YearLocator()
            )
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter("%Y")
            )

        elif frequency in {"W", "D"}:
            locator = mdates.AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(locator)
            )

        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        return _save_figure_atomic(fig, filename)

    finally:
        plt.close(fig)
