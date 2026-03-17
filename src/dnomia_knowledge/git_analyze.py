"""Git history analysis: crossover classification and thresholds."""

from __future__ import annotations

import statistics


def classify_file(churn: int, reads: int, churn_p75: float, reads_p75: float) -> str:
    """Classify a file based on churn and read activity.

    Returns one of: BLIND, TURBULENT, STABLE, HOT, ZOMBIE, COLD.
    """
    high_churn = churn >= churn_p75 and churn_p75 > 0
    high_reads = reads >= reads_p75 and reads_p75 > 0
    zero_churn = churn == 0
    zero_reads = reads == 0

    if high_churn and zero_reads:
        return "BLIND"
    if high_churn and high_reads:
        return "HOT"
    if high_churn and not high_reads and not zero_reads:
        return "TURBULENT"
    if zero_churn and reads > 0:
        return "ZOMBIE"
    if not high_churn and high_reads:
        return "STABLE"
    return "COLD"


def _percentile_75(values: list[int]) -> float:
    """Calculate 75th percentile. Returns 0 if empty."""
    if not values:
        return 0
    sorted_vals = sorted(values)
    if len(sorted_vals) < 4:
        return max(sorted_vals)
    return statistics.quantiles(sorted_vals, n=4)[2]  # Q3 = 75th percentile


def classify_crossover_results(rows: list[dict]) -> list[dict]:
    """Add 'signal' classification to crossover query results.

    Computes P75 thresholds from the data itself.
    """
    if not rows:
        return []

    churn_values = [r["churn"] for r in rows]
    read_values = [r["reads"] for r in rows]

    churn_p75 = _percentile_75(churn_values)
    reads_p75 = _percentile_75(read_values)

    result = []
    for row in rows:
        classified = dict(row)
        classified["signal"] = classify_file(row["churn"], row["reads"], churn_p75, reads_p75)
        result.append(classified)

    return result
