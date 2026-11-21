from pathlib import Path

from pocket_filter.motion_intervals import load_motion_interval_catalog, lookup_motion_interval


def test_motion_interval_lookup_by_filename(tmp_path: Path):
    poses = tmp_path / "6" / "sess" / "user_1" / "poses_full.npy"
    poses.parent.mkdir(parents=True)
    poses.touch()
    cat = tmp_path / "motion.jsonl"
    cat.write_text(
        '{"path": "6/sess/user_1/poses_full.npy", "t0": 1.0, "t1": 2.5, "score": 0.9}\n',
        encoding="utf-8",
    )
    catalog = load_motion_interval_catalog(cat)
    iv = lookup_motion_interval(poses, catalog)
    assert iv is not None
    assert iv.t0 == 1.0 and iv.t1 == 2.5 and iv.score == 0.9