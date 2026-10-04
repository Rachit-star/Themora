from app.detectors import duplicates, invalid_values, drift, leakage

DETECTORS = [duplicates, invalid_values, drift, leakage]
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def run_all(df, target=None, ref=None):
    findings = []
    for detector in DETECTORS:
        findings += detector.detect(df, target=target, ref=ref)
    return sorted(findings, key=lambda f: SEVERITY_ORDER[f.severity])