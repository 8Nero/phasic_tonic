import numpy as np
import pytest
import warnings
from scipy.signal import hilbert, windows
from neurodsp.filt import filter_signal

from phasic_tonic.core import (
    preprocess_rem_epoch,
    get_phasic_candidates,
    is_valid_phasic,
    compute_thresholds,
    get_sequences,
    get_segments,
    get_rem_epochs,
    get_start_end,
    detect_troughs,
    smooth_signal,
)

from phasic_tonic.detect import detect_phasic

def sim_rem_epochs(fs, num_epochs=3, epoch_length=10):
    rem_epochs = {}
    t_total = 0
    for i in range(num_epochs):
        t = np.arange(0, epoch_length, 1/fs)
        signal = np.sin(2 * np.pi * 8 * t) + np.random.normal(0, 1, size=t.shape)
        
        start_time = int(t_total)
        end_time = int(t_total + epoch_length)
        
        rem_epochs[(start_time, end_time)] = signal
        
        # Increment total time with a random gap between epochs
        t_total += epoch_length + np.random.uniform(1, 10)

    return rem_epochs

def test_preprocess_rem_epoch():
    fs=500
    rem_epochs = sim_rem_epochs(fs, num_epochs=1, epoch_length=10)
    epoch = next(iter(rem_epochs.values()))
    
    inst_phase, inst_amp = preprocess_rem_epoch(epoch, fs)
    
    # Check that outputs are numpy arrays
    assert isinstance(inst_phase, np.ndarray)
    assert isinstance(inst_amp, np.ndarray)
    
    # Check that outputs have the same shape as input epoch
    assert inst_phase.shape == epoch.shape
    assert inst_amp.shape == epoch.shape
    
    # Check that phase values are within the expected range
    assert np.all(inst_phase >= -np.pi)
    assert np.all(inst_phase <= np.pi)
    
    # Check that instantaneous amplitude is non-negative
    assert np.all(inst_amp >= 0)

def test_get_phasic_candidates():
    smoothed_trough_differences = np.array([0.1, 0.05, 0.2, 0.03, 0.15])
    trough_indices = np.array([10, 20, 30, 40, 50, 60])
    threshold_percentile_10 = 0.1
    thr_dur = 50  # in milliseconds
    fs = 100
    
    candidates = get_phasic_candidates(
        smoothed_trough_differences,
        trough_indices,
        threshold_percentile_10,
        thr_dur,
        fs,
    )
    
    expected_candidates = [(0, 2), (3, 4)]
    assert candidates == expected_candidates

def test_is_valid_phasic():
    smoothed_diffs_slice = np.array([0.03, 0.04, 0.02])
    inst_amp_slice = np.array([0.5, 0.6, 0.7])
    threshold_percentile_5 = 0.05
    mean_amplitude_threshold = 0.6
    
    # Expected result is True
    result = is_valid_phasic(
        smoothed_diffs_slice,
        inst_amp_slice,
        threshold_percentile_5,
        mean_amplitude_threshold,
    )
    assert result == True
    
    # Test with mean amplitude below threshold
    inst_amp_slice = np.array([0.5, 0.4, 0.3])
    result = is_valid_phasic(
        smoothed_diffs_slice,
        inst_amp_slice,
        threshold_percentile_5,
        mean_amplitude_threshold,
    )
    assert result == False

def test_compute_thresholds():
    fs=1000
    rem_epochs = sim_rem_epochs(fs, num_epochs=3, epoch_length=10)
    
    # Call the function
    thresholds, epoch_trough_idx, epoch_smooth_diffs, epoch_amplitudes = compute_thresholds(rem_epochs, fs)
    
    # Check that thresholds is a tuple of three floats
    assert isinstance(thresholds, tuple)
    assert len(thresholds) == 3
    assert all(isinstance(t, float) for t in thresholds)
    
    # Check that returned dictionaries are not empty
    assert len(epoch_trough_idx) == len(rem_epochs)
    assert len(epoch_smooth_diffs) == len(rem_epochs)
    assert len(epoch_amplitudes) == len(rem_epochs)
    
    # Test ValueError when rem_epochs is empty
    with pytest.raises(ValueError):
        compute_thresholds({}, fs)

def test_get_sequences():
    a = np.array([1, 2, 3, 5, 6, 9])
    expected_sequences = [(1, 3), (5, 6), (9, 9)]
    
    sequences = get_sequences(a)

    assert sequences == expected_sequences

def test_get_segments():
    idx = [(0, 2), (4, 6)]
    signal = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    expected_segments = [np.array([0, 1]), np.array([4, 5])]
    
    segments = get_segments(idx, signal)

    for seg, exp_seg in zip(segments, expected_segments):
        assert np.array_equal(seg, exp_seg)

def test_get_rem_epochs():
    # Sample EEG and hypnogram data
    eeg = np.arange(1000)
    hypno = np.zeros(100)
    hypno[20:50] = 5
    fs = 10
    rem_epochs = get_rem_epochs(eeg, hypno, fs, min_dur=3)
    
    # Expected REM epoch keys
    assert len(rem_epochs) == 1
    assert (20, 49) in rem_epochs
    
    # Expected EEG segment
    expected_epoch = eeg[20 * fs : (49 + 1) * fs]
    actual_epoch = rem_epochs[(20, 49)]
    assert np.array_equal(actual_epoch, expected_epoch)
    
    # Test ValueError when no REM epochs longer than min_dur
    hypno = np.zeros(100)
    hypno[0:6] = 5
    with pytest.raises(ValueError):
        get_rem_epochs(eeg, hypno, fs, min_dur=10)

def test_get_start_end():
    sleep_states = np.array([1, 1, 2, 2, 2, 1, 1, 3, 3, 1])
    sleep_state_id = 2
    start, end = get_start_end(sleep_states, sleep_state_id)
    assert start == [2]
    assert end == [4]

def test_detect_troughs():
    signal = np.array([0, -1, 0, -2, 0, -1, 0])
    threshold = -1.5
    troughs = detect_troughs(signal, threshold)
    expected_troughs = np.array([3])
    assert np.array_equal(troughs, expected_troughs)

def test_smooth_signal():
    signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    window_size = 3
    smoothed = smooth_signal(signal, window_size)
    filt = np.ones(window_size) / window_size
    expected_smoothed = np.convolve(signal, filt, 'same')
    assert np.allclose(smoothed, expected_smoothed)

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