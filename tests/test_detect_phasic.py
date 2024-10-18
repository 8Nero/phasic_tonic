import numpy as np
import pytest
import warnings

from phasic_tonic.detect import detect_phasic

def test_detect_phasic():
    fs = 1000
    eeg_length = 200*fs
    eeg = np.sin(2 * np.pi * 8 * eeg_length) + np.random.normal(0, 0.1, size=eeg_length)

    hypno_length = 200
    hypno = np.zeros(hypno_length)
    for (start, end) in [(10,30), (50,55), (170,171)]:
        hypno[start:end] = 5

    # Test EEG longer than hypnogram
    eeg_long = np.concatenate([eeg, np.zeros(int(fs * 10))])  # Extend EEG by 10 seconds
    with warnings.catch_warnings(record=True) as w:
        detect_phasic(eeg_long, hypno, fs)
        # Check that a warning was issued
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "EEG is longer than hypnogram" in str(w[-1].message)

    # Test EEG shorter than hypnogram
    eeg_short = eeg[:len(eeg) - int(fs * 10)]  # Shorten EEG by 10 seconds
    with pytest.raises(ValueError) as excinfo:
        detect_phasic(eeg_short, hypno, fs)
        assert "EEG is shorter than hypnogram" in str(excinfo.value)

    # Test invalid EEG type
    with pytest.raises(ValueError) as excinfo:
        detect_phasic("invalid_eeg", hypno, fs)
        assert "EEG must be a numpy array." in str(excinfo.value)

    # Test invalid hypnogram type
    with pytest.raises(ValueError) as excinfo:
        phasic_rem_periods_invalid_hypno = detect_phasic(eeg, "invalid_hypno", fs)
        assert "Hypnogram must be a numpy array." in str(excinfo.value)