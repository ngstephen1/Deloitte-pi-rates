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
        self.required_columns = [
            "case_id",
            "transactionid",
            "accountid",
            "decision",
            "analyst_notes",
            "created_at",
            "updated_at",
            "review_version",
        ]
        self.decisions = self._load_decisions()

    def _load_decisions(self) -> pd.DataFrame:
        """Load existing decisions from file, or create empty DataFrame."""
        if self.storage_file.exists():
            try:
                df = pd.read_csv(self.storage_file)
                for column in self.required_columns:
                    if column not in df.columns:
                        df[column] = "" if column not in {"review_version"} else 1
                if "timestamp" in df.columns:
                    df["updated_at"] = df["updated_at"].replace("", pd.NA).fillna(df["timestamp"])
                    df["created_at"] = df["created_at"].replace("", pd.NA).fillna(df["timestamp"])
                df["case_id"] = df["case_id"].replace("", pd.NA).fillna(df["transactionid"])
                df["decision"] = df["decision"].map(self._normalize_decision)
                df["review_version"] = pd.to_numeric(df["review_version"], errors="coerce").fillna(1).astype(int)
                df = df[self.required_columns]
                LOGGER.info(f"Loaded {len(df)} existing analyst decisions from {self.storage_file}")
                self._ensure_storage_file(df)
                return df
            except Exception as e:
                LOGGER.warning(f"Could not load decisions file: {e}; starting fresh")

        # Create empty DataFrame with schema
        empty_df = pd.DataFrame(columns=self.required_columns)
        self._ensure_storage_file(empty_df)
        return empty_df

    def _normalize_decision(self, decision: str) -> str:
        mapping = {
            "Approve": "Approve Flag",
            "Approve Flag": "Approve Flag",
            "Dismiss": "Dismiss",
            "Needs Review": "Needs Review",
        }
        return mapping.get(str(decision), str(decision))

    def _ensure_storage_file(self, df: pd.DataFrame) -> None:
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_file.exists():
            df.to_csv(self.storage_file, index=False)

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
        normalized_decision = self._normalize_decision(decision)
        if normalized_decision not in config.DECISION_OPTIONS:
            raise ValueError(f"Invalid decision: {decision}. Must be one of {config.DECISION_OPTIONS}")

        # Check if transaction already has a decision
        existing = self.decisions[self.decisions["transactionid"] == transaction_id]
        now = datetime.now().isoformat()

        new_record = pd.DataFrame({
            "case_id": [transaction_id],
            "transactionid": [transaction_id],
            "accountid": [account_id],
            "decision": [normalized_decision],
            "analyst_notes": [notes or ""],
            "created_at": [now],
            "updated_at": [now],
            "review_version": [1],
        })

        if len(existing) > 0:
            # Update existing
            idx = existing.index[0]
            current_version = int(self.decisions.at[idx, "review_version"]) if pd.notna(self.decisions.at[idx, "review_version"]) else 1
            self.decisions.at[idx, "case_id"] = transaction_id
            self.decisions.at[idx, "decision"] = normalized_decision
            self.decisions.at[idx, "analyst_notes"] = notes or ""
            self.decisions.at[idx, "accountid"] = account_id
            self.decisions.at[idx, "updated_at"] = now
            if pd.isna(self.decisions.at[idx, "created_at"]) or self.decisions.at[idx, "created_at"] == "":
                self.decisions.at[idx, "created_at"] = now
            self.decisions.at[idx, "review_version"] = current_version + 1
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
            "case_id": row["case_id"],
            "transactionid": row["transactionid"],
            "accountid": row["accountid"],
            "decision": row["decision"],
            "analyst_notes": row["analyst_notes"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "review_version": row["review_version"],
        }

    def get_all_decisions(self) -> pd.DataFrame:
        """Get all recorded decisions."""
        decisions = self.decisions.copy()
        if "updated_at" in decisions.columns:
            decisions = decisions.sort_values("updated_at", ascending=False, na_position="last").reset_index(drop=True)
        return decisions

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
        self.decisions = self.decisions[self.required_columns].copy()
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
    store.record_decision("TX000001", "AC00001", "Approve Flag", "Normal transaction pattern")
    store.record_decision("TX000002", "AC00002", "Needs Review", "Unusual amount for account")
    store.record_decision("TX000003", "AC00003", "Dismiss", "Expected merchant")

    print(store.summary_statistics())
    print(store.get_all_decisions())
