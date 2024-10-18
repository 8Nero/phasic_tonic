import pytest
import numpy as np
import pandas as pd
from unittest import mock
from unittest.mock import MagicMock

from pynapple import IntervalSet

from phasic_tonic.analysis import PhasicTonic

@pytest.fixture
def phasic_tonic_instance():
    return PhasicTonic(fs=100, thr_dur=900)

def test_detect_eeg_length_trim(
    phasic_tonic_instance
):
    fs = phasic_tonic_instance.fs
    hypno = np.array([5]*10)  # 10 epochs
    eeg_length = 1000  # Greater than 10 * fs = 1000
    eeg = np.random.randn(eeg_length + 100)  # Longer by 100 samples

    with mock.patch('warnings.warn') as mock_warn:
        result = phasic_tonic_instance.detect(eeg, hypno)
        mock_warn.assert_called_once_with("EEG is longer than hypnogram. Trimming EEG to match hypnogram length.")
    
    assert len(result) == 2
    assert result["phasic_intervals"] is not None
    assert result["tonic_intervals"] is not None
    
    # EEG should be trimmed
    assert len(phasic_tonic_instance.rem_intervals.start) == 1

def test_detect_eeg_length_short(
    phasic_tonic_instance
):
    fs = phasic_tonic_instance.fs
    hypno = np.array([5]*10)  # 10 epochs
    eeg_length = 900  # Less than 10 * fs = 1000
    eeg = np.random.randn(eeg_length)

    with pytest.raises(ValueError, match="EEG is shorter than hypnogram. Please ensure EEG and hypnogram lengths match."):
        phasic_tonic_instance.detect(eeg, hypno)

def test_detect_valid(
    phasic_tonic_instance
):
    fs = phasic_tonic_instance.fs
    hypno = np.array([5]*10)  # 10 epochs
    eeg_length = 1000  # Exactly 10 * fs
    eeg = np.random.randn(eeg_length)

    result = phasic_tonic_instance.detect(eeg, hypno)

    assert "phasic_intervals" in result
    assert "tonic_intervals" in result

    # Assuming rem_intervals is set correctly
    assert phasic_tonic_instance.rem_intervals.start == 0
    assert phasic_tonic_instance.rem_intervals.end == 9

def test_compute_stats(phasic_tonic_instance):
    # Manually set rem_intervals, phasic_intervals, tonic_intervals
    phasic_tonic_instance.rem_intervals = IntervalSet([0], [10])
    phasic_tonic_instance.phasic_intervals = IntervalSet([1, 3], [2, 4])
    phasic_tonic_instance.tonic_intervals = IntervalSet([0, 5], [1, 10])

    stats_df = phasic_tonic_instance.compute_stats()

    # Expected DataFrame structure
    assert isinstance(stats_df, pd.DataFrame)
    expected_columns = [
        "rem_start", "rem_end", "state",
        "num_bouts", "mean_duration",
        "total_duration", "percent_of_rem"
    ]
    assert list(stats_df.columns) == expected_columns

    # Check values
    # rem_start and rem_end should be integers
    assert stats_df["rem_start"].iloc[0] == 0
    assert stats_df["rem_end"].iloc[0] == 10

    # Check states
    assert set(stats_df["state"]) == {"phasic", "tonic"}

    # Check num_bouts
    phasic_bouts = stats_df[stats_df["state"] == "phasic"]["num_bouts"].iloc[0]
    tonic_bouts = stats_df[stats_df["state"] == "tonic"]["num_bouts"].iloc[0]
    assert phasic_bouts == 2
    assert tonic_bouts == 2

    # Check durations
    phasic_total = (2-1) + (4-3)
    tonic_total = (1-0) + (10-5)
    assert stats_df[stats_df["state"] == "phasic"]["total_duration"].iloc[0] == phasic_total
    assert stats_df[stats_df["state"] == "tonic"]["total_duration"].iloc[0] == tonic_total

    # Check percent_of_rem
    assert stats_df[stats_df["state"] == "phasic"]["percent_of_rem"].iloc[0] == (phasic_total / 10) * 100
    assert stats_df[stats_df["state"] == "tonic"]["percent_of_rem"].iloc[0] == (tonic_total / 10) * 100
