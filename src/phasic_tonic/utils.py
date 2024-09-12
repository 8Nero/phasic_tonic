import numpy as np

import yasa
from mne.filter import resample

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger('runtime')

def get_sequences(a: np.ndarray, ibreak: int = 1) -> List[Tuple[int, int]]:
    """
    Identify contiguous sequences.

    Parameters
    ----------
    a : array_like
        Input array.
    ibreak : int, optional
        Threshold value for determining breaks between sequences, by default 1.

    Returns
    -------
    List[Tuple[int, int]]
        List of tuples containing the start and end integer of each contiguous sequence.
    """
    if len(a) == 0:
        return []

    diff = np.diff(a)
    breaks = np.where(diff > ibreak)[0]
    breaks = np.append(breaks, len(a) - 1)
    
    sequences = []
    start_idx = 0
    
    for break_idx in breaks:
        end_idx = break_idx
        sequences.append((a[start_idx], a[end_idx]))
        start_idx = end_idx + 1
    
    return sequences

def get_segments(idx: List[Tuple[int, int]], signal: np.ndarray) -> List[np.ndarray]:
    """
    Extract segments of the signal between specified start and end time indices.

    Parameters
    ----------
    idx : List[Tuple[int, int]]
        List of tuples, each containing (start_time, end_time).
    signal : np.ndarray
        The signal from which to extract segments.

    Returns
    -------
    List[np.ndarray]
        List of signal segments corresponding to the given time ranges.
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

    Parameters
    ----------
    eeg : np.ndarray
        EEG signal.
    hypno : np.ndarray
        Hypnogram array.
    fs : float
        Sampling frequency.
    min_dur : float, optional
        Minimum duration of REM epoch in seconds, by default 3.

    Returns
    -------
    Dict[Tuple[int, int], np.ndarray]
        Dictionary of REM epochs with sequence indices as keys.

    Raises
    ------
    ValueError
        If no REM epochs greater than min_dur are found.
    """
    rem_seq = get_sequences(np.where(hypno == 5)[0])
    rem_idx = [(start * fs, (end + 1) * fs) for start, end in rem_seq if (end - start) > min_dur]
   
    if not rem_idx:
        raise ValueError("No REM epochs greater than min_dur.")
   
    rem_epochs = get_segments(rem_idx, eeg)
    return {seq: seg for seq, seg in zip(rem_seq, rem_epochs)}

def create_hypnogram(phasicREM, length):
       binary_hypnogram = np.zeros(length, dtype=int)
       for start, end in phasicREM:
           binary_hypnogram[start:end] = 1
       return binary_hypnogram
   
def get_start_end(sleep_states, sleep_state_id, fs=500):
    seq = get_sequences(np.where(sleep_states==sleep_state_id)[0])
    start, end = [], []
    for s, e in seq:
        start.append(s)
        end.append(e)
    return (start, end)

def preprocess(signal: np.ndarray, n_down: int, target_fs=500) -> np.ndarray:
    """Downsample and remove artifacts."""
    # Downsample to 500 Hz
    data = resample(signal, down=n_down, method='fft', npad='auto')
    # Remove artifacts
    art_std, _ = yasa.art_detect(data, target_fs , window=1, method='std', threshold=4)
    art_up = yasa.hypno_upsample_to_data(art_std, 1, data, target_fs)
    data[art_up] = 0
    data -= data.mean()
    return data

def _despine_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
