import json
import sys

import pandas as pd

from app.detectors import run_all


def main():
    if len(sys.argv) < 2:
        print("Usage: python audit.py data.csv [target_column]")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    target = sys.argv[2] if len(sys.argv) > 2 else None

    findings = run_all(df, target=target)
    print(json.dumps([f.to_dict() for f in findings], indent=2))


if __name__ == "__main__":
    main()