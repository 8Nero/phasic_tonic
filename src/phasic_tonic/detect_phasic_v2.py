"""
Phasic REM Detection Module

Algorithm described by Mizuseki, K., Diba, K., Pastalkova, E. et al. 
Hippocampal CA1 pyramidal cells form functionally distinct sublayers. Nat Neurosci 14, 1174–1181 (2011).
https://doi.org/10.1038/nn.2894

Dependencies:
- numpy
- scipy
- neurodsp
"""

import numpy as np
from scipy.signal import hilbert
from neurodsp.filt import filter_signal
from typing import Dict, List, Tuple, Union

def get_sequences(x, ibreak=1):
    """
    Identifies contiguous sequences.

    Args:
      x (np.ndarray): 1D time series.
      ibreak (int): A threshold value for determining breaks between sequences (default is 1).

    Returns:
        list of tuples: Each tuple contains the start and end integer of each contiguous sequence.
    """
    if len(x) == 0:
        return []

    diff = np.diff(x)
    breaks = np.where(diff > ibreak)[0]

    # Append the last index to handle the end of the array
    breaks = np.append(breaks, len(x) - 1)
    
    sequences = []
    start_idx = 0
    
    for break_idx in breaks:
        end_idx = break_idx
        sequences.append((x[start_idx], x[end_idx]))
        start_idx = end_idx + 1
    
    return sequences

def get_segments(idx, signal):
    """
    Extracts segments of the signal between specified start and end time indices.

    Args:
        idx (list of tuples): Each tuple contains (start_time, end_time).
        signal (np.ndarray): The signal from which to extract segments.

    Returns:
        list of np.ndarray: Each element is a segment of the signal corresponding to the given time ranges.
    """
    segments = []
    for (start_time, end_time) in idx:
        if end_time > len(signal):
            end_time = len(signal) - 1
        segment = signal[start_time:end_time]
        segments.append(segment)
    
    return segments

def get_rem_epochs(eeg: np.ndarray, hypno: np.ndarray, fs: float, min_dur: float = 3) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Extract REM epochs from EEG data based on hypnogram.

    Args:
        eeg (np.ndarray): EEG signal.
        hypno (np.ndarray): Hypnogram array.
        fs (float): Sampling frequency.
        min_dur (float): Minimum duration of REM epoch in seconds.

    Returns:
        Dict[Tuple[int, int], np.ndarray]: Dictionary of REM epochs with sequence indices as keys.

    Raises:
        ValueError: If no REM epochs greater than min_dur are found.
    """
    rem_seq = get_sequences(np.where(hypno == 5)[0])
    rem_idx = [(start * fs, (end + 1) * fs) for start, end in rem_seq if (end - start) > min_dur]
   
    if not rem_idx:
        raise ValueError("No REM epochs greater than min_dur.")
   
    rem_epochs = get_segments(rem_idx, eeg)
    return {seq: seg for seq, seg in zip(rem_seq, rem_epochs)}

def preprocess_rem_epoch(epoch: np.ndarray, fs: float, w1: float = 5.0, w2: float = 12.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess a REM epoch by applying bandpass filter and Hilbert transform.

    Args:
        epoch (np.ndarray): REM epoch signal.
        fs (float): Sampling frequency.
        w1 (float): Lower frequency bound for bandpass filter.
        w2 (float): Upper frequency bound for bandpass filter.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Instantaneous phase and amplitude of the processed signal.
    """
    epoch = filter_signal(epoch, fs, 'bandpass', (w1, w2), remove_edges=False)
    analytic_signal = hilbert(epoch)
    return np.angle(analytic_signal), np.abs(analytic_signal)

def detect_troughs(signal: np.ndarray, threshold: float = -3) -> np.ndarray:
    """
    Detect troughs in a signal.
    """
    lidx = np.where(signal[0:-2] > signal[1:-1])[0]
    ridx = np.where(signal[1:-1] <= signal[2:])[0]
    thidx = np.where(signal[1:-1] < threshold)[0]
    return np.intersect1d(lidx, np.intersect1d(ridx, thidx)) + 1

def smooth_signal(signal: np.ndarray, window_size: int = 11) -> np.ndarray:
    """
    Apply moving average smoothing to a signal.
    """
    filt = np.ones(window_size) / window_size
    return np.convolve(signal, filt, 'same')

def get_phasic_candidates(sdiff: np.ndarray, tridx: np.ndarray, thr1: float, thr_dur: float, fs: float) -> List[Tuple[int, int]]:
    """
    Get candidate phasic REM periods based on smoothed trough differences.
    """
    cand_idx = np.where(sdiff <= thr1)[0]
    cand = get_sequences(cand_idx)
    return [(start, end) for start, end in cand if ((tridx[end] - tridx[start] + 1) / fs) * 1000 >= thr_dur]

def is_valid_phasic(start: int, end: int, sdiff: np.ndarray, eegh: np.ndarray, tridx: np.ndarray, thr2: float, thr3: float) -> bool:
    """
    Check if a candidate phasic REM period is valid.

    Returns:
        bool: True if the candidate is a valid phasic REM period, False otherwise.
    """
    min_sdiff = np.min(sdiff[start:end])
    mean_amp = np.mean(eegh[tridx[start]:tridx[end]+1])
    return min_sdiff <= thr2 and mean_amp >= thr3

def compute_thresholds(rem_epochs, fs):
    """
    Computes thresholds for detecting phasic REM.
    """
    trough_difference_list = []
    rem_eeg = np.array([])
    eeg_seq, smooth_difference_seq, trough_idx_seq = {}, {}, {}
   
    for idx, epoch in rem_epochs.items():
        inst_phase, inst_amp = preprocess_rem_epoch(epoch, fs)
        
        # trough indices
        trough_idx = detect_troughs(inst_phase)

        # trough differences
        trough_difference = np.diff(trough_idx)
       
        # smoothed trough differences
        smooth_difference_seq[idx] = smooth_signal(trough_difference)
        trough_idx_seq[idx] = trough_idx
        eeg_seq[idx] = inst_amp
       
        # differences between troughs
        trough_difference_list.extend(trough_difference)

        # amplitude of the entire REM sleep
        rem_eeg = np.concatenate((rem_eeg, inst_amp))
   
    trough_difference_smooth = smooth_signal(np.array(trough_difference_list))
    thresholds = (np.percentile(trough_difference_smooth, 10), np.percentile(trough_difference_smooth, 5), rem_eeg.mean())
    return thresholds, trough_idx_seq, smooth_difference_seq, eeg_seq
    


def detect_phasic(eeg: np.ndarray, hypno: np.ndarray, fs: float, thr_dur: float = 900) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """
    Detect phasic REM periods in EEG data.

    Args:
        eeg (np.ndarray): EEG signal.
        hypno (np.ndarray): Hypnogram array.
        fs (float): Sampling frequency.
        thr_dur (float): Minimum duration threshold for phasic REM in milliseconds.

    Returns:
        Dict[Tuple[int, int], List[Tuple[int, int]]]: Dictionary of detected phasic REM periods for each REM epoch.
    """
    rem_epochs = get_rem_epochs(eeg, hypno, fs)
    
    thresholds, trough_idx_seq, smooth_difference_seq, eeg_seq = compute_thresholds(rem_epochs, fs)
    thr1, thr2, thr3 = thresholds

    #   Optional Plotting function here
    #   ...
    #

    phasicREM = {rem_idx: [] for rem_idx in rem_epochs.keys()}
   
    for rem_idx, trough_idx in trough_idx_seq.items():
        rem_start, rem_end = rem_idx
        offset = rem_start * fs
        smooth_difference, eegh = smooth_difference_seq[rem_idx], eeg_seq[rem_idx]
       
        candidates = get_phasic_candidates(smooth_difference, trough_idx, thr1, thr_dur, fs)
       
        for start, end in candidates:
            if is_valid_phasic(start, end, smooth_difference, eegh, trough_idx, thr2, thr3):
                t_a = trough_idx[start] + offset
                t_b = min(trough_idx[end] + offset, rem_end * fs)
                phasicREM[rem_idx].append((t_a, t_b + 1))
   
    return phasicREM

def detect_phasic_v2(rem_epochs:Dict[Tuple[int, int], np.ndarray], fs: float, thr_dur: float = 900) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """
    Detect phasic REM periods in EEG data.

    Args:
        rem_epochs (Dict[Tuple[int, int], np.ndarray]): Dictionary of REM epochs with sequence indices as keys.
        fs (float): Sampling frequency.
        thr_dur (float): Minimum duration threshold for phasic REM in milliseconds.

    Returns:
        Dict[Tuple[int, int], List[Tuple[int, int]]]: Dictionary of detected phasic REM periods for each REM epoch.
    """

    thresholds, trough_idx_seq, smooth_difference_seq, eeg_seq = compute_thresholds(rem_epochs, fs)
    thr1, thr2, thr3 = thresholds

    phasicREM = {rem_idx: [] for rem_idx in rem_epochs.keys()}
   
    for rem_idx, trough_idx in trough_idx_seq.items():
        rem_start, rem_end = rem_idx
        offset = rem_start * fs
        smooth_difference, eegh = smooth_difference_seq[rem_idx], eeg_seq[rem_idx]
       
        candidates = get_phasic_candidates(smooth_difference, trough_idx, thr1, thr_dur, fs)
       
        for start, end in candidates:
            if is_valid_phasic(start, end, smooth_difference, eegh, trough_idx, thr2, thr3):
                t_a = trough_idx[start] + offset
                t_b = min(trough_idx[end] + offset, rem_end * fs)
                phasicREM[rem_idx].append((t_a, t_b + 1))
   
    return phasicREM