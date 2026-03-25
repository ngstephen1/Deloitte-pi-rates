"""
Analyst review decisions storage module.
Simple CSV-based storage for analyst decisions (can be upgraded to SQLite later).
"""

from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd

from . import config
from .utils import LOGGER


class ReviewStore:
    """
    Manages analyst review decisions and notes.
    Stores decisions in CSV format for transparency and portability.
    """

    def __init__(self, storage_file: Path = None):
        """
        Initialize review store.

        Args:
            storage_file: Path to CSV for storing decisions (default: config.ANALYST_DECISIONS_FILE)
        """
        self.storage_file = storage_file or config.ANALYST_DECISIONS_FILE
        self.decisions = self._load_decisions()

    def _load_decisions(self) -> pd.DataFrame:
        """Load existing decisions from file, or create empty DataFrame."""
        if self.storage_file.exists():
            try:
                df = pd.read_csv(self.storage_file)
                LOGGER.info(f"Loaded {len(df)} existing analyst decisions from {self.storage_file}")
                return df
            except Exception as e:
                LOGGER.warning(f"Could not load decisions file: {e}; starting fresh")

        # Create empty DataFrame with schema
        return pd.DataFrame(columns=[
            "transactionid",
            "accountid",
            "decision",
            "analyst_notes",
            "timestamp",
        ])

    def record_decision(
        self,
        transaction_id: str,
        account_id: str,
        decision: str,
        notes: str = "",
    ) -> None:
        """
        Record an analyst decision for a transaction.

        Args:
            transaction_id: Transaction ID
            account_id: Account ID
            decision: One of config.DECISION_OPTIONS (Approve, Dismiss, Needs Review)
            notes: Optional analyst notes
        """
        if decision not in config.DECISION_OPTIONS:
            raise ValueError(f"Invalid decision: {decision}. Must be one of {config.DECISION_OPTIONS}")

        # Check if transaction already has a decision
        existing = self.decisions[self.decisions["transactionid"] == transaction_id]

        new_record = pd.DataFrame({
            "transactionid": [transaction_id],
            "accountid": [account_id],
            "decision": [decision],
            "analyst_notes": [notes or ""],
            "timestamp": [datetime.now().isoformat()],
        })

        if len(existing) > 0:
            # Update existing
            idx = existing.index[0]
            self.decisions.at[idx, "decision"] = decision
            self.decisions.at[idx, "analyst_notes"] = notes or ""
            self.decisions.at[idx, "timestamp"] = datetime.now().isoformat()
        else:
            # Add new
            self.decisions = pd.concat([self.decisions, new_record], ignore_index=True)

        self._save_decisions()

    def get_decision(self, transaction_id: str) -> Optional[dict]:
        """
        Retrieve recorded decision for a transaction, or None if not recorded.

        Returns:
            Dictionary with decision info, or None
        """
        result = self.decisions[self.decisions["transactionid"] == transaction_id]

        if len(result) == 0:
            return None

        row = result.iloc[0]
        return {
            "transactionid": row["transactionid"],
            "accountid": row["accountid"],
            "decision": row["decision"],
            "analyst_notes": row["analyst_notes"],
            "timestamp": row["timestamp"],
        }

    def get_all_decisions(self) -> pd.DataFrame:
        """Get all recorded decisions."""
        return self.decisions.copy()

    def get_decisions_by_status(self, decision_type: str) -> pd.DataFrame:
        """
        Get all decisions of a given type.

        Args:
            decision_type: One of config.DECISION_OPTIONS

        Returns:
            Filtered DataFrame
        """
        return self.decisions[self.decisions["decision"] == decision_type].copy()

    def summary_statistics(self) -> dict:
        """
        Return summary statistics of analyst decisions so far.

        Returns:
            Dictionary with decision counts and percentages
        """
        total = len(self.decisions)

        if total == 0:
            return {
                "total_decisions": 0,
                "by_type": {decision: 0 for decision in config.DECISION_OPTIONS},
            }

        summary = {
            "total_decisions": total,
            "by_type": {},
        }

        for decision_type in config.DECISION_OPTIONS:
            count = (self.decisions["decision"] == decision_type).sum()
            pct = 100 * count / total
            summary["by_type"][decision_type] = {
                "count": count,
                "percentage": pct,
            }

        return summary

    def _save_decisions(self) -> None:
        """Save decisions to CSV."""
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        self.decisions.to_csv(self.storage_file, index=False)
        LOGGER.debug(f"Saved {len(self.decisions)} decisions to {self.storage_file}")

    def export_decision_summary(self, output_file: Path = None) -> None:
        """Export decision summary as CSV."""
        if output_file is None:
            output_file = config.REPORTS_DIR / "analyst_decision_summary.csv"

        summary_data = []
        for decision_type in config.DECISION_OPTIONS:
            count = (self.decisions["decision"] == decision_type).sum()
            summary_data.append({
                "decision_type": decision_type,
                "count": count,
                "percentage": 100 * count / len(self.decisions) if len(self.decisions) > 0 else 0,
            })

        summary_df = pd.DataFrame(summary_data)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_file, index=False)
        LOGGER.info(f"Exported decision summary to {output_file}")


if __name__ == "__main__":
    # For testing
    store = ReviewStore()

    # Record some sample decisions
    store.record_decision("TX000001", "AC00001", "Approve", "Normal transaction pattern")
    store.record_decision("TX000002", "AC00002", "Needs Review", "Unusual amount for account")
    store.record_decision("TX000003", "AC00003", "Dismiss", "Expected merchant")

    print(store.summary_statistics())
    print(store.get_all_decisions())
