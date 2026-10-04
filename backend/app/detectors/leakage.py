"""
Leakage & suspicious-feature detector.

Three checks:
1. **Target leakage** – features with suspiciously high correlation /
   association to the target column (likely derived from the label).
2. **Near-constant columns** – columns where 99 %+ of values are the same
   (zero information, but can hide bugs).
3. **ID-like columns** – columns where almost every value is unique,
   which are likely row identifiers and shouldn't be used as features.
"""

import numpy as np
from app.models import Finding

CORR_THRESHOLD = 0.95          # absolute Pearson r above which we flag
NEAR_CONSTANT_FRAC = 0.99      # fraction of most common value to flag
UNIQUE_FRAC = 0.95             # fraction of unique values to flag as ID-like
MIN_ROWS = 10


def detect(df, target=None, ref=None):
    findings = []
    findings += _target_leakage(df, target)
    findings += _near_constant(df)
    findings += _id_like_columns(df)
    return findings


# ── Target leakage ───────────────────────────────────────────────────────────

def _target_leakage(df, target):
    """Flag features with very high correlation to the target."""
    if target is None or target not in df.columns:
        return []

    findings = []
    numeric = df.select_dtypes("number")
    if target not in numeric.columns:
        return []

    target_series = numeric[target].dropna()
    if len(target_series) < MIN_ROWS:
        return []

    for col in numeric.columns:
        if col == target:
            continue
        col_series = numeric[col].dropna()
        # Align indices
        common = target_series.index.intersection(col_series.index)
        if len(common) < MIN_ROWS:
            continue

        r = float(np.corrcoef(target_series.loc[common], col_series.loc[common])[0, 1])
        if np.isnan(r):
            continue
        if abs(r) >= CORR_THRESHOLD:
            findings.append(
                Finding(
                    detector="leakage",
                    severity="high",
                    column=col,
                    title=f"'{col}' is {abs(r):.0%} correlated with target '{target}'",
                    evidence={
                        "pearson_r": round(r, 4),
                        "target": target,
                        "why_it_matters": (
                            "A feature this strongly correlated with the target "
                            "is likely derived from it (data leakage) and will "
                            "cause inflated metrics that don't generalise."
                        ),
                    },
                )
            )
    return findings


# ── Near-constant columns ────────────────────────────────────────────────────

def _near_constant(df):
    """Flag columns where almost every row has the same value."""
    if len(df) < MIN_ROWS:
        return []
    findings = []
    for col in df.columns:
        top_freq = df[col].value_counts(normalize=True, dropna=False)
        if len(top_freq) == 0:
            continue
        dominant_frac = float(top_freq.iloc[0])
        if dominant_frac >= NEAR_CONSTANT_FRAC:
            dominant_value = top_freq.index[0]
            findings.append(
                Finding(
                    detector="leakage",
                    severity="low",
                    column=col,
                    title=f"'{col}' is near-constant ({dominant_frac:.1%} = {dominant_value!r})",
                    evidence={
                        "dominant_value": str(dominant_value),
                        "dominant_fraction": round(dominant_frac, 4),
                        "unique_values": int(df[col].nunique(dropna=False)),
                        "why_it_matters": (
                            "A near-constant feature carries almost no information "
                            "and can mask bugs or silent data issues."
                        ),
                    },
                )
            )
    return findings


# ── ID-like columns ──────────────────────────────────────────────────────────

def _id_like_columns(df):
    """Flag columns where almost every value is unique (row IDs, UUIDs, etc.)."""
    if len(df) < MIN_ROWS:
        return []
    findings = []
    for col in df.columns:
        n_unique = df[col].nunique(dropna=True)
        frac = n_unique / len(df)
        if frac >= UNIQUE_FRAC and n_unique > MIN_ROWS:
            # Exclude numeric columns that look like continuous features
            if df[col].dtype in ("float64", "float32"):
                continue
            findings.append(
                Finding(
                    detector="leakage",
                    severity="medium",
                    column=col,
                    title=f"'{col}' looks like a row identifier ({frac:.0%} unique)",
                    evidence={
                        "unique_fraction": round(frac, 4),
                        "n_unique": n_unique,
                        "n_rows": len(df),
                        "sample_values": [str(v) for v in df[col].dropna().head(5).tolist()],
                        "why_it_matters": (
                            "Row identifiers should not be used as ML features. "
                            "They can cause overfitting to specific records."
                        ),
                    },
                )
            )
    return findings
