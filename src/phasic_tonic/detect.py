"""
This module provides the `detect_phasic` function for detecting phasic REM sleep epochs in EEG data.
"""
from typing import Dict, List, Tuple
import numpy as np
import warnings

from .core import get_rem_epochs, compute_thresholds, get_phasic_candidates, is_valid_phasic


def detect_phasic(
    eeg: np.ndarray, 
    hypno: np.ndarray, 
    fs: int, 
    thr_dur: int = 900
) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """
    Detect phasic REM epochs in EEG data based on the method described by Mizuseki et al. (2011).
    
    Parameters
    ----------
    eeg : np.ndarray
        EEG signal array.
    hypno : np.ndarray
        Hypnogram array. Expects an array of 1-second epochs where REM stage corresponds to value '5'.
    fs : int
        Sampling rate, in Hz.
    thr_dur : int, optional
        Minimum duration threshold for a phasic REM epoch in milliseconds, by default 900.

    Returns
    -------
    Dict[Tuple[int, int], List[Tuple[int, int]]]
        Dictionary where keys are tuples indicating the start and end times of REM epochs (in seconds),
        and values are list of tuples indicating the start and end times of detected phasic REM epochs (in samples).

    Raises
    ------
    ValueError
        If `eeg` or `hypno` is not a NumPy array.
        If the length of `eeg` does not match the expected length based on `hypno` and `fs`.

    Warnings
    --------
    Warns if the EEG signal is longer than the hypnogram and trims it to match the hypnogram length.
    """
    if not isinstance(eeg, np.ndarray):
        raise ValueError("EEG must be a numpy array.")
    if not isinstance(hypno, np.ndarray):
        raise ValueError("Hypnogram must be a numpy array.")

    # Hypnogram should be 1-second epochs
    expected_eeg_length = len(hypno) * fs
    if len(eeg) > expected_eeg_length:
        warnings.warn("EEG is longer than hypnogram. Trimming EEG to match hypnogram length.")
        eeg = eeg[:int(expected_eeg_length)]
    elif len(eeg) < expected_eeg_length:
        raise ValueError("EEG is shorter than hypnogram. Please ensure EEG and hypnogram lengths match.")
    
    rem_epochs = get_rem_epochs(eeg, hypno, fs)
    thresholds, epoch_trough_idx, epoch_smooth_diffs, epoch_amplitudes = compute_thresholds(rem_epochs, fs)
    threshold_percentile_10, threshold_percentile_5, mean_amplitude_threshold = thresholds

    phasicREM = {}

    for rem_idx, trough_indices in epoch_trough_idx.items():
        rem_start, rem_end = rem_idx
        offset = rem_start * fs
        smooth_difference = epoch_smooth_diffs[rem_idx]
        inst_amp = epoch_amplitudes[rem_idx]

        # Get candidate periods
        candidates = get_phasic_candidates(smooth_difference, trough_indices, threshold_percentile_10, thr_dur, fs)

        # Validate candidates
        valid_periods = []
        for start, end in candidates:
            smoothed_diffs_slice = smooth_difference[start:end]
            inst_amp_slice = inst_amp[trough_indices[start]:trough_indices[end] + 1]

            if is_valid_phasic(smoothed_diffs_slice, inst_amp_slice, threshold_percentile_5, mean_amplitude_threshold):
                start_time = trough_indices[start] + offset
                end_time = min(trough_indices[end] + offset, rem_end * fs)
                valid_periods.append((int(start_time), int(end_time) + 1))

        if valid_periods:
            phasicREM[rem_idx] = valid_periods

    return phasicREM
