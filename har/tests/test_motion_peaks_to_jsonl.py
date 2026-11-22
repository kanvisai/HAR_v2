import csv
from pathlib import Path

from motion_peaks_to_jsonl import from_batch_csv


def test_from_batch_oracle_intervals(tmp_path: Path):
    csv_path = tmp_path / "batch.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["path", "category", "eval_interval_t0", "eval_interval_t1", "error"],
        )
        w.writeheader()
        w.writerow(
            {
                "path": "/data/data_result/6/sess1/user_a/poses_full.npy",
                "category": "6",
                "eval_interval_t0": "1.0",
                "eval_interval_t1": "2.5",
                "error": "",
            }
        )
    recs = from_batch_csv(csv_path, data_root=None, min_duration=0.35, use_motion_cols=False)
    assert len(recs) == 1
    assert recs[0]["path"].endswith("6/sess1/user_a/poses_full.npy")
    assert recs[0]["t0"] == 1.0