"""
Benford's Law analysis for transaction amounts.
Detects systematic anomalies and potential fraud patterns by comparing
observed first-digit distributions to Benford's expected distribution.
"""

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .utils import LOGGER, save_csv


def get_first_digit(value: float) -> int:
    """Extract first significant digit from a number."""
    if value <= 0:
        return None
    # Remove decimal point and leading zeros to get first significant digit
    digits_only = str(abs(value)).replace(".", "").replace("-", "").lstrip("0")
    if not digits_only:
        return None
    first = int(digits_only[0])
    # Benford's Law only applies to digits 1-9
    if first == 0:
        return None
    return first


def benford_expected_distribution() -> dict:
    """
    Return expected first-digit distribution according to Benford's Law.
    P(d) = log10(1 + 1/d) for d in [1, 9]
    """
    return {
        d: np.log10(1 + 1 / d)
        for d in range(1, 10)
    }


def compute_benford_statistic(amounts: pd.Series) -> Tuple[dict, dict, float]:
    """
    Compute observed vs expected first-digit distributions.

    Args:
        amounts: Series of transaction amounts

    Returns:
        Tuple of (observed_dist, expected_dist, chi_squared_statistic)
    """
    # Extract first digits
    first_digits = amounts.apply(get_first_digit).dropna()

    if len(first_digits) == 0:
        LOGGER.warning("No valid transaction amounts for Benford analysis")
        return {}, benford_expected_distribution(), np.nan

    # Count observed distribution
    observed_counts = first_digits.value_counts().sort_index()
    observed_dist = (observed_counts / observed_counts.sum()).to_dict()

    # Ensure all digits 1-9 are present
    for d in range(1, 10):
        if d not in observed_dist:
            observed_dist[d] = 0

    # Expected distribution
    expected_dist = benford_expected_distribution()

    # Chi-squared statistic
    chi_squared = 0
    for d in range(1, 10):
        observed = observed_counts.get(d, 0)
        expected = expected_dist[d] * observed_counts.sum()
        if expected > 0:
            chi_squared += (observed - expected) ** 2 / expected

    return observed_dist, expected_dist, chi_squared


def flag_benford_anomalies(df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """
    Flag transactions with first digits that deviate significantly from Benford's Law.

    Args:
        df: DataFrame with 'transactionamount' column
        threshold: Percentile below which a transaction is flagged (e.g., 0.05 for bottom 5%)

    Returns:
        DataFrame with 'benford_anomaly_flag' column added
    """
    first_digits = df["transactionamount"].apply(get_first_digit).dropna()

    if len(first_digits) == 0:
        df["benford_anomaly_flag"] = False
        return df

    # Get observed distribution
    observed_counts = first_digits.value_counts().sort_index()
    observed_dist = (observed_counts / observed_counts.sum()).to_dict()

    # Get expected distribution
    expected_dist = benford_expected_distribution()

    # Calculate deviation for each digit
    digit_deviations = {}
    for d in range(1, 10):
        obs = observed_dist.get(d, 0)
        exp = expected_dist[d]
        deviation = abs(obs - exp)
        digit_deviations[d] = deviation

    # Find digits with highest deviation
    high_deviation_threshold = np.percentile(list(digit_deviations.values()), 100 * (1 - threshold))

    # Flag transactions with unusual first digits
    def is_anomaly(amount):
        digit = get_first_digit(amount)
        if digit is None:
            return False
        # Safe access - if digit not in dictionary, it's not anomalous
        if digit not in digit_deviations:
            return False
        return digit_deviations[digit] > high_deviation_threshold

    df["benford_anomaly_flag"] = df["transactionamount"].apply(is_anomaly).astype(int)

    return df


def benford_analysis(df: pd.DataFrame) -> dict:
    """
    Run full Benford's Law analysis.

    Returns:
        Dictionary with analysis results and summary
    """
    LOGGER.info("Running Benford's Law analysis on transaction amounts...")

    # Ensure transactionamount exists
    if "transactionamount" not in df.columns:
        LOGGER.warning("transactionamount column not found; skipping Benford analysis")
        return {"status": "skipped", "reason": "missing_column"}

    # Filter positive amounts
    positive_amounts = df[df["transactionamount"] > 0]["transactionamount"]

    if len(positive_amounts) == 0:
        LOGGER.warning("No positive transaction amounts found")
        return {"status": "skipped", "reason": "no_positive_amounts"}

    # Compute distributions
    observed_dist, expected_dist, chi_squared = compute_benford_statistic(positive_amounts)

    # Flag anomalies
    df = flag_benford_anomalies(df)

    anomaly_count = (df["benford_anomaly_flag"] == 1).sum()

    results = {
        "status": "complete",
        "total_transactions": len(df),
        "transactions_analyzed": len(positive_amounts),
        "anomalies_flagged": anomaly_count,
        "anomaly_percentage": 100 * anomaly_count / len(df),
        "chi_squared": chi_squared,
        "observed_distribution": observed_dist,
        "expected_distribution": expected_dist,
    }

    LOGGER.info(f"  Analyzed {len(positive_amounts)} positive amounts")
    LOGGER.info(f"  Chi-squared statistic: {chi_squared:.4f}")
    LOGGER.info(f"  Flagged {anomaly_count} ({results['anomaly_percentage']:.2f}%) as anomalous")

    return results, df


def plot_benford_distribution(
    observed_dist: dict,
    expected_dist: dict,
    output_path: Path = None,
) -> None:
    """
    Plot observed vs expected Benford's Law distributions.
    """
    if output_path is None:
        output_path = config.FIGURES_DIR / "benford_law_analysis.png"

    digits = list(range(1, 10))
    observed = [observed_dist.get(d, 0) for d in digits]
    expected = [expected_dist[d] for d in digits]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(digits))
    width = 0.35

    ax.bar(x - width / 2, observed, width, label="Observed", alpha=0.8)
    ax.bar(x + width / 2, expected, width, label="Expected (Benford)", alpha=0.8)

    ax.set_xlabel("First Digit", fontsize=12)
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_title("Transaction Amount Distribution: Benford's Law Analysis", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(digits)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.FIGURE_DPI)
    plt.close()

    LOGGER.info(f"Saved Benford's Law plot to {output_path}")


if __name__ == "__main__":
    # Test with sample data
    sample_amounts = pd.Series([10.5, 250.0, 1500.0, 5.25, 75.0, 300.0, 45.0])
    obs, exp, chi2 = compute_benford_statistic(sample_amounts)
    print(f"Chi-squared: {chi2}")
    print(f"Observed: {obs}")
    print(f"Expected: {exp}")
