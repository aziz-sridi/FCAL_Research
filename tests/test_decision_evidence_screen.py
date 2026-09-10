import json

import pytest

from fcal_study.decision_evidence_screen import _history, _slug


def test_v2_and_v2_plus_never_share_an_evaluation_directory():
    names = ["CE", "FCAL V1", "FCAL V2", "FCAL V2+", "CAMAL", "GAIN-ext",
             "RRR", "Faster-ISNet", "V2 class-mean"]
    assert len({_slug(name) for name in names}) == len(names)


def test_incomplete_training_is_not_compared_with_fifty_epoch_runs(tmp_path):
    rows = [{"epoch": epoch, "train": {"seconds": 10, "peak_cuda_memory_bytes": 100},
             "validation": {"seconds": 1, "accuracy": .8}} for epoch in range(1, 51)]
    history = tmp_path / "history.jsonl"
    history.write_text("\n".join(json.dumps(row) for row in rows))
    result = _history(tmp_path)
    assert result["post_branch_seconds"] == 45 * 11
    assert result["steady_epoch_median_seconds"] == 11
    history.write_text("\n".join(json.dumps(row) for row in rows[:-1]))
    with pytest.raises(ValueError, match="Incomplete training history"):
        _history(tmp_path)
