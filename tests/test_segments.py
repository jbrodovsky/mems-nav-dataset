"""Tests for splitting a recording at IMU dropouts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from memsnav.preprocess import INIT_COLUMNS, orphan_source, segment_label, split_on_imu_gaps


def _recording(seconds: int, rate_hz: int = 10, gaps: list[tuple[float, float]] = ()):
    index = pd.date_range("2025-01-01", periods=seconds * rate_hz, freq=f"{1000 // rate_hz}ms")
    frame = pd.DataFrame(1.0, index=index, columns=list(INIT_COLUMNS))
    elapsed = (index - index[0]).total_seconds()
    for start, end in gaps:
        frame.loc[(elapsed >= start) & (elapsed < end), ["acc_x", "gyro_x"]] = np.nan
    return frame


def test_labels_run_like_spreadsheet_columns():
    assert [segment_label(i) for i in (0, 1, 25, 26, 27)] == ["A", "B", "Z", "AA", "AB"]


def test_an_unbroken_recording_is_one_unlabelled_segment():
    segments = split_on_imu_gaps(_recording(600))
    assert [(s.label, s.status) for s in segments] == [("", "kept")]


def test_a_long_dropout_splits_and_a_short_one_does_not():
    segments = split_on_imu_gaps(_recording(1200, gaps=[(100, 103), (600, 620)]))
    assert [(s.label, s.status) for s in segments] == [("A", "kept"), ("B", "kept")]
    assert segments[0].end_s < 600 <= segments[1].start_s


def test_a_fragment_under_the_minimum_is_reported_not_kept():
    segments = split_on_imu_gaps(_recording(700, gaps=[(100, 120)]))
    assert [(s.label, s.status) for s in segments] == [
        ("A", "shorter than the 300 s minimum"),
        ("B", "kept"),
    ]
    assert segments[0].data.empty


def test_orphans_trace_back_to_their_recording():
    sources = {"2025-11-09_17-34-01"}
    assert orphan_source("2025-11-09_17-34-01_B", sources) == "2025-11-09_17-34-01"
    assert orphan_source("2025-11-09_17-34-01", sources) == "2025-11-09_17-34-01"
    assert orphan_source("2025-11-10_00-00-00", sources) is None
