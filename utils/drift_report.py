from pathlib import Path
from typing import Optional, Sequence
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import ColumnDriftMetric

def generate_drift_report(
    ref_path: Path,
    cur_path: Optional[Path],
    out_path: Path,
    ref_frac: float = 0.90,                 # reference = first 90%
    tests_cols: Optional[Sequence[str]] = None,
    stattest: str = "wasserstein",          # numeric test
    stattest_threshold: float = 0.05
) -> Path:
    """
    Build an Evidently drift report.
    - If cur_path is None, split ref_path by ref_frac: first part = reference, last = current.
    - tests_cols: optional list of important columns to add stricter ColumnDriftMetric for.
    """
    df = pd.read_csv(ref_path)

    if cur_path is not None and Path(cur_path).exists():
        ref_df = df
        cur_df = pd.read_csv(cur_path)
    else:
        n = len(df)
        split = int(n * ref_frac) if n > 0 else 0
        ref_df = df.iloc[:split].copy()
        cur_df = df.iloc[split:].copy()

    metrics = [DataDriftPreset()]
    if tests_cols:
        for c in tests_cols:
            if c in df.columns:
                metrics.append(
                    ColumnDriftMetric(
                        column_name=c,
                        stattest=stattest,
                        stattest_threshold=stattest_threshold
                    )
                )

    report = Report(metrics=metrics)
    report.run(reference_data=ref_df, current_data=cur_df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out_path))
    return out_path

if __name__ == "__main__":
    import sys
    ref = Path(sys.argv[1]) if len(sys.argv) >= 2 else Path("data/processed/features_aug.csv")
    cur = Path(sys.argv[2]) if len(sys.argv) >= 3 else None
    out = Path(sys.argv[3]) if len(sys.argv) >= 4 else Path("reports/evidently_drift.html")
    p = generate_drift_report(ref, cur, out)
    print("Wrote:", p)
