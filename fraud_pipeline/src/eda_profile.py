"""
Exploratory Data Analysis and Profiling module.
Generates summary statistics, visualizations, and exports for Tableau.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .benford import benford_analysis, plot_benford_distribution
from .utils import LOGGER, save_csv, detect_basic_outliers, check_missing_values


def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for numeric columns.

    Returns:
        DataFrame with statistics (count, mean, std, min, 25%, 50%, 75%, max)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    stats = df[numeric_cols].describe().T
    stats["null_count"] = df[numeric_cols].isnull().sum()
    stats["null_pct"] = 100 * df[numeric_cols].isnull().sum() / len(df)

    return stats


def compute_categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary for categorical columns: unique counts and mode.
    """
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    summaries = []
    for col in categorical_cols:
        unique_count = df[col].nunique()
        mode_series = df[col].mode()
        mode_val = mode_series.iloc[0] if len(mode_series) > 0 else None
        mode_freq = (df[col] == mode_val).sum() if mode_val is not None else 0

        summaries.append({
            "column": col,
            "unique_values": unique_count,
            "mode": mode_val,
            "mode_frequency": mode_freq,
            "null_count": df[col].isnull().sum(),
        })

    return pd.DataFrame(summaries)


def detect_outliers_in_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect outliers in all numeric columns using IQR method.

    Returns:
        DataFrame with columns: column, outlier_count, outlier_pct, bounds
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    results = []

    for col in numeric_cols:
        outlier_mask, stats = detect_basic_outliers(df[col], method="iqr")

        if stats:
            results.append({
                "column": col,
                "outlier_count": stats["outlier_count"],
                "outlier_pct": 100 * stats["outlier_count"] / len(df),
                "lower_bound": stats["lower_bound"],
                "upper_bound": stats["upper_bound"],
                "Q1": stats["Q1"],
                "Q3": stats["Q3"],
                "IQR": stats["IQR"],
            })

    return pd.DataFrame(results)


def plot_transaction_amount_distribution(df: pd.DataFrame, output_path: Path = None) -> None:
    """Plot distribution of transaction amounts."""
    if output_path is None:
        output_path = config.FIGURES_DIR / "transaction_amount_distribution.png"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(df["transactionamount"], bins=50, color=config.COLOR_NORMAL, alpha=0.7, edgecolor="black")
    axes[0].set_xlabel("Transaction Amount ($)", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].set_title("Transaction Amount Distribution", fontsize=12, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    # Box plot
    axes[1].boxplot(df["transactionamount"], vert=True)
    axes[1].set_ylabel("Transaction Amount ($)", fontsize=11)
    axes[1].set_title("Transaction Amount Box Plot", fontsize=12, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.FIGURE_DPI)
    plt.close()

    LOGGER.info(f"Saved transaction amount plot to {output_path}")


def plot_transaction_type_breakdown(df: pd.DataFrame, output_path: Path = None) -> None:
    """Plot breakdown of transaction types."""
    if output_path is None:
        output_path = config.FIGURES_DIR / "transaction_type_breakdown.png"

    type_counts = df["transactiontype"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(type_counts, labels=type_counts.index, autopct="%1.1f%%", startangle=90)
    ax.set_title("Transaction Type Distribution", fontsize=12, fontweight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.FIGURE_DPI)
    plt.close()

    LOGGER.info(f"Saved transaction type plot to {output_path}")


def plot_channel_breakdown(df: pd.DataFrame, output_path: Path = None) -> None:
    """Plot breakdown by transaction channel."""
    if output_path is None:
        output_path = config.FIGURES_DIR / "channel_breakdown.png"

    channel_counts = df["channel"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    channel_counts.plot(kind="barh", ax=ax, color=config.COLOR_NORMAL, alpha=0.7)
    ax.set_xlabel("Number of Transactions", fontsize=11)
    ax.set_title("Transactions by Channel", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.FIGURE_DPI)
    plt.close()

    LOGGER.info(f"Saved channel breakdown plot to {output_path}")


def plot_customer_demographics(df: pd.DataFrame, output_path: Path = None) -> None:
    """Plot customer age and occupation distributions."""
    if output_path is None:
        output_path = config.FIGURES_DIR / "customer_demographics.png"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Age distribution
    axes[0].hist(df["customerage"], bins=30, color=config.COLOR_NORMAL, alpha=0.7, edgecolor="black")
    axes[0].set_xlabel("Customer Age", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].set_title("Customer Age Distribution", fontsize=12, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    # Occupation distribution
    occupation_counts = df["customeroccupation"].value_counts()
    occupation_counts.plot(kind="barh", ax=axes[1], color=config.COLOR_NORMAL, alpha=0.7)
    axes[1].set_xlabel("Number of Customers", fontsize=11)
    axes[1].set_title("Customers by Occupation", fontsize=12, fontweight="bold")
    axes[1].grid(axis="x", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.FIGURE_DPI)
    plt.close()

    LOGGER.info(f"Saved demographics plot to {output_path}")


def plot_login_attempts_distribution(df: pd.DataFrame, output_path: Path = None) -> None:
    """Plot login attempts distribution (potential anomaly signal)."""
    if output_path is None:
        output_path = config.FIGURES_DIR / "login_attempts_distribution.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df["loginattempts"], bins=20, color=config.COLOR_NORMAL, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Login Attempts", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Distribution of Login Attempts Before Transaction", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Mark high values
    high_threshold = df["loginattempts"].quantile(0.95)
    ax.axvline(high_threshold, color=config.COLOR_ANOMALOUS, linestyle="--", linewidth=2, label=f"95th percentile: {high_threshold}")
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.FIGURE_DPI)
    plt.close()

    LOGGER.info(f"Saved login attempts plot to {output_path}")


def eda_and_profile(df: pd.DataFrame, output_dir: Path = None) -> dict:
    """
    Run full EDA and profiling pipeline.

    Args:
        df: Cleaned transaction DataFrame
        output_dir: Directory to save outputs (default: config.REPORTS_DIR)

    Returns:
        Dictionary with analysis results
    """
    if output_dir is None:
        output_dir = config.REPORTS_DIR

    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("STAGE 2: EXPLORATORY ANALYSIS & PROFILING")
    LOGGER.info("=" * 60)

    # Summary statistics
    LOGGER.info("Computing summary statistics...")
    numeric_stats = compute_summary_statistics(df)
    save_csv(numeric_stats.reset_index(), output_dir / "numeric_summary_statistics.csv")

    categorical_summary = compute_categorical_summary(df)
    save_csv(categorical_summary, output_dir / "categorical_summary.csv")

    # Missing values
    LOGGER.info("Checking missing values...")
    missing = check_missing_values(df)
    if len(missing) > 0:
        save_csv(missing, output_dir / "missing_values_summary.csv")
    else:
        LOGGER.info("  No missing values found")

    # Outliers
    LOGGER.info("Detecting outliers...")
    outliers = detect_outliers_in_dataset(df)
    save_csv(outliers, output_dir / "outliers_summary.csv")

    # Benford's Law
    benford_results, df_with_benford = benford_analysis(df)
    plot_benford_distribution(
        benford_results.get("observed_distribution", {}),
        benford_results.get("expected_distribution", {}),
    )

    # Visualizations
    LOGGER.info("Creating visualizations...")
    plot_transaction_amount_distribution(df)
    plot_transaction_type_breakdown(df)
    plot_channel_breakdown(df)
    plot_customer_demographics(df)
    plot_login_attempts_distribution(df)

    # Create top locations summary
    top_locations = df["location"].value_counts().head(20).reset_index()
    top_locations.columns = ["location", "transaction_count"]
    save_csv(top_locations, output_dir / "top_locations.csv")

    LOGGER.info(f"\nStage 2 complete. Outputs saved to {output_dir}\n")

    return {
        "numeric_stats": numeric_stats,
        "categorical_summary": categorical_summary,
        "missing_values": missing,
        "outliers": outliers,
        "benford_results": benford_results,
    }


if __name__ == "__main__":
    # For testing: load cleaned data and run EDA
    from .ingest_clean import load_and_clean

    df = load_and_clean()
    results = eda_and_profile(df)
    print("EDA complete")
