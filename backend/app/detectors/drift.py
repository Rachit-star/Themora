"""
Drift detector -- flags columns whose distribution shifted between
a reference dataset and the current dataset.

If no reference DataFrame is supplied, the current data is split in half
(first rows vs last rows) as a simple proxy for temporal drift.

Numeric columns  -> two-sample Kolmogorov-Smirnov test
Categorical cols -> Jensen-Shannon divergence on value counts
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from app.models import Finding

KS_P_THRESHOLD = 0.01          # p-value below which we flag numeric drift
JS_THRESHOLD = 0.15            # JS divergence above which we flag categorical drift
MIN_ROWS = 30                  # need enough rows per split


def detect(df, target=None, ref=None):
    """Return a list of Finding objects for columns that exhibit drift."""
    if ref is not None:
        return _compare(ref, df, label="ref -> current")
    # No reference: split current dataset in half as a proxy
    if len(df) < MIN_ROWS * 2:
        return []
    mid = len(df) // 2
    return _compare(df.iloc[:mid], df.iloc[mid:], label="first-half -> second-half")


def _compare(df_a, df_b, label):
    findings = []
    findings += _numeric_drift(df_a, df_b, label)
    findings += _categorical_drift(df_a, df_b, label)
    return findings


# -- Numeric drift (KS test) --------------------------------------------------

def _numeric_drift(df_a, df_b, label):
    findings = []
    shared = [c for c in df_a.select_dtypes("number").columns if c in df_b.columns]
    for col in shared:
        a = df_a[col].dropna().values
        b = df_b[col].dropna().values
        if len(a) < MIN_ROWS or len(b) < MIN_ROWS:
            continue
        stat, p = stats.ks_2samp(a, b)
        if p < KS_P_THRESHOLD:
            severity = "high" if p < KS_P_THRESHOLD / 10 else "medium"
            findings.append(
                Finding(
                    detector="drift",
                    severity=severity,
                    column=col,
                    title=f"Distribution shift in '{col}' ({label})",
                    evidence={
                        "test": "Kolmogorov-Smirnov",
                        "ks_statistic": round(float(stat), 4),
                        "p_value": round(float(p), 6),
                        "mean_a": round(float(np.mean(a)), 4),
                        "mean_b": round(float(np.mean(b)), 4),
                    },
                )
            )
    return findings


# -- Categorical drift (Jensen-Shannon divergence) ----------------------------

def _categorical_drift(df_a, df_b, label):
    findings = []
    cat_cols = [
        c for c in df_a.select_dtypes(include=["object", "category", "bool"]).columns
        if c in df_b.columns
    ]
    for col in cat_cols:
        a_counts = df_a[col].value_counts(normalize=True)
        b_counts = df_b[col].value_counts(normalize=True)
        # Align on the union of categories
        all_cats = sorted(set(a_counts.index) | set(b_counts.index))
        p = np.array([a_counts.get(c, 0.0) for c in all_cats])
        q = np.array([b_counts.get(c, 0.0) for c in all_cats])

        # scipy returns sqrt(JSD), square it to get actual divergence
        jsd = float(jensenshannon(p, q) ** 2)
        if jsd < JS_THRESHOLD:
            continue

        severity = "high" if jsd > 0.35 else "medium"
        findings.append(
            Finding(
                detector="drift",
                severity=severity,
                column=col,
                title=f"Category distribution shift in '{col}' ({label})",
                evidence={
                    "test": "Jensen-Shannon divergence",
                    "js_divergence": round(jsd, 4),
                    "top_categories_a": dict(a_counts.head(5)),
                    "top_categories_b": dict(b_counts.head(5)),
                },
            )
        )
    return findings
