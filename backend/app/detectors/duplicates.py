from app.models import Finding

HIGH_SEVERITY_FRACTION = 0.05


def detect(df, target=None, ref=None):
    if len(df) == 0:
        return []
    is_duplicate = df.duplicated()
    count = int(is_duplicate.sum())

    if count == 0:
        return []

    fraction = count / len(df)
    severity = "high" if fraction > HIGH_SEVERITY_FRACTION else "medium"

    return [
        Finding(
            detector="duplicates",
            severity=severity,
            column=None,
            title=f"{count} exact duplicate rows ({fraction:.1%})",
            evidence={
                "count": count,
                "fraction": round(fraction, 4),
                "example_row_indices": [int(i) for i in df[is_duplicate].index[:5]],
                "why_it_matters": "Copies of a row can land in both train and test sets and inflate scores.",
            },
        )
    ]