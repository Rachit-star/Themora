from app.models import Finding

MISSING_CUTOFF = 0.2
HIGH_MISSING_CUTOFF = 0.5
MAD_Z_CUTOFF = 6
MIN_ROWS_FOR_STATS = 20


def detect(df, target=None, ref=None):
    return _missing_values(df) + _extreme_values(df)


def _missing_values(df):
    findings = []
    for col in df.columns:
        fraction = float(df[col].isna().mean())
        if fraction < MISSING_CUTOFF:
            continue
        findings.append(
            Finding(
                detector="missing_values",
                severity="high" if fraction >= HIGH_MISSING_CUTOFF else "medium",
                column=col,
                title=f"{fraction:.0%} of '{col}' is missing",
                evidence={"missing_fraction": round(fraction, 4)},
            )
        )
    return findings


def _extreme_values(df):
    findings = []
    for col in df.select_dtypes("number").columns:
        values = df[col].dropna()
        if len(values) < MIN_ROWS_FOR_STATS:
            continue

        median = values.median()
        mad = (values - median).abs().median()
        if mad == 0:
            continue

        z = 0.6745 * (values - median) / mad
        extreme = values[z.abs() > MAD_Z_CUTOFF]
        normal = values[z.abs() <= MAD_Z_CUTOFF]
        if len(extreme) == 0:
            continue

        findings.append(
            Finding(
                detector="invalid_values",
                severity="medium",
                column=col,
                title=f"{len(extreme)} extreme value(s) in '{col}'",
                evidence={
                    "count": int(len(extreme)),
                    "examples": [float(v) for v in extreme.head(5)],
                    "typical_range_1st_to_99th_percentile": [
                        float(values.quantile(0.01)),
                        float(values.quantile(0.99)),
                    ],
                    "method": f"modified z-score above {MAD_Z_CUTOFF} (median and MAD)",
                },
            )
        )
    return findings