"""
This module provides the `PhasicTonic` class for detecting phasic and tonic REM sleep periods in EEG data.
"""

import numpy as np
import pandas as pd
import pynapple as nap
import warnings

from .core import compute_thresholds, get_rem_epochs, get_phasic_candidates, is_valid_phasic, get_start_end

class PhasicTonic:
    """
    A class for detecting phasic and tonic REM sleep periods in EEG data.

    The `PhasicTonic` class provides methods to identify and analyze phasic and tonic substates within REM sleep epochs based on EEG signals.
    It implements threshold-based algorithms to differentiate detect phasic REM periods.

    Attributes
    ----------
    fs : int
        Sampling frequency of the EEG data in Hz.
    thr_dur : float
        Threshold duration for phasic REM detection in milliseconds.
    rem_intervals : pynapple.IntervalSet or None
        IntervalSet representing the detected REM sleep periods.
    phasic_intervals : pynapple.IntervalSet or None
        IntervalSet representing the detected phasic REM periods.
    tonic_intervals : pynapple.IntervalSet or None
        IntervalSet representing the detected tonic REM periods.
    thresholds : Tuple[float, float, float] or None
        Tuple containing:

            - 10th percentile of smoothed trough differences across all epochs.
            - 5th percentile of smoothed trough differences across all epochs.
            - Mean instantaneous amplitude across all REM epochs.
    epoch_trough_idx : Dict[Tuple[int, int], np.ndarray] or None
        Dictionary mapping REM epoch indices to their trough indices.
    epoch_smooth_diffs : Dict[Tuple[int, int], np.ndarray] or None
        Dictionary mapping REM epoch indices to their smoothed trough differences.
    epoch_amplitudes : Dict[Tuple[int, int], np.ndarray] or None
        Dictionary mapping REM epoch indices to their instantaneous amplitudes.
    """

    def __init__(self, fs: int, thr_dur: float = 900):
        """
        Initialize the PhasicTonic.

        Parameters
        ----------
        fs : int
            Sampling frequency of the EEG data.
        thr_dur : float, optional
            Threshold duration for phasic REM detection in milliseconds, by default 900.
        """
        self.fs = fs
        self.thr_dur = thr_dur
        self.rem_intervals = None
        self.phasic_intervals = None
        self.tonic_intervals = None
        self.thresholds = None
        self.epoch_trough_idx = None
        self.epoch_smooth_diffs = None
        self.epoch_amplitudes = None

    def detect(self, eeg: np.ndarray, hypno: np.ndarray):
        """
        Detect phasic and tonic REM periods based on EEG data and a hypnogram.

        This method processes the EEG signal alongside a hypnogram to identify REM sleep periods.
        It then applies threshold-based algorithms to classify segments of REM sleep as either phasic or tonic.

        Parameters
        ----------
        eeg : np.ndarray
            EEG signal array. Should be a 1D NumPy array representing the EEG amplitude over time.
        hypno : np.ndarray
            Hypnogram array. Each element represents a sleep stage for a corresponding epoch.
            Assumes that the value `5` corresponds to REM sleep.

        Returns
        -------
        dict
            A dictionary containing:
                
                'phasic_intervals': pynapple.IntervalSet of detected phasic REM periods.
                'tonic_intervals': pynapple.IntervalSet of detected tonic REM periods.

        Raises
        ------
        ValueError
            If `eeg` or `hypno` is not a NumPy array.
            If the length of `eeg` does not match the expected length based on `hypno` and `fs`.

        Warnings
        --------
        Warns if the EEG signal is longer than the hypnogram and trims it to match the hypnogram length.

        Examples
        --------
        >>> pt = PhasicTonic(fs=1000)
        >>> result = pt.detect(eeg_signal, hypnogram)
        >>> phasic = result['phasic_intervals']
        >>> tonic = result['tonic_intervals']
        """
        if not isinstance(eeg, np.ndarray):
            raise ValueError("EEG must be a numpy array.")
        if not isinstance(hypno, np.ndarray):
            raise ValueError("Hypnogram must be a numpy array.")

        # Hypnogram should be 1-second epochs
        expected_eeg_length = len(hypno) * self.fs
        if len(eeg) > expected_eeg_length:
            warnings.warn("EEG is longer than hypnogram. Trimming EEG to match hypnogram length.")
            eeg = eeg[:int(expected_eeg_length)]
        elif len(eeg) < expected_eeg_length:
            raise ValueError("EEG is shorter than hypnogram. Please ensure EEG and hypnogram lengths match.")
        
        # REM states are assumed to be mapped as 5
        rem_start, rem_end = get_start_end(hypno, 5)
        # Create IntervalSet for REM intervals
        self.rem_intervals = nap.IntervalSet(rem_start, rem_end)
        
        # Extract REM epochs from EEG
        rem_epochs = get_rem_epochs(eeg, hypno, self.fs)
        # Compute thresholds and intermediate values
        self.thresholds, self.epoch_trough_idx, self.epoch_smooth_diffs, self.epoch_amplitudes = compute_thresholds(rem_epochs, self.fs)

        threshold_10th, threshold_5th, mean_inst_amp = self.thresholds

        phasic_rem = {}
        for rem_idx, trough_indices in self.epoch_trough_idx.items():
            rem_start, rem_end = rem_idx

            # Offset to align indices within the original EEG signal
            offset = rem_start * self.fs
            smooth_difference = self.epoch_smooth_diffs[rem_idx]
            inst_amp = self.epoch_amplitudes[rem_idx]

            candidates = get_phasic_candidates(
                smooth_difference, trough_indices, threshold_10th, self.thr_dur, self.fs
            )

            valid_periods = []
            for start, end in candidates:
                if is_valid_phasic(
                    smooth_difference[start:end],
                    inst_amp[trough_indices[start]:trough_indices[end] + 1],
                    threshold_5th,
                    mean_inst_amp
                ):
                    start_time = trough_indices[start] + offset
                    end_time = min(trough_indices[end] + offset, rem_end * self.fs)
                    valid_periods.append((int(start_time), int(end_time) + 1))

            if valid_periods:
                phasic_rem[rem_idx] = valid_periods

        self._create_interval_sets(phasic_rem)
        return {
            "phasic_intervals": self.phasic_intervals,
            "tonic_intervals": self.tonic_intervals
        }

    def _create_interval_sets(self, phasic_rem: dict):
        """
        Create intervals for phasic and tonic REM periods.
        """
        ph_start, ph_end = [], []
        for intervals in phasic_rem.values():
            for start, end in intervals:
                ph_start.append(start / self.fs)
                ph_end.append(end / self.fs)
        
        # Create an IntervalSet for phasic periods using start and end times in seconds
        self.phasic_intervals = nap.IntervalSet(ph_start, ph_end)
        # Compute tonic intervals by subtracting phasic intervals from REM intervals
        self.tonic_intervals = self.rem_intervals.set_diff(self.phasic_intervals)

    def compute_stats(self) -> pd.DataFrame:
        """
        Compute statistics for phasic and tonic REM periods.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the following columns:

                - 'rem_start': Start time of the REM epoch in seconds.
                - 'rem_end': End time of the REM epoch in seconds.
                - 'state': Type of REM state ('phasic' or 'tonic').
                - 'num_bouts': Number of distinct phasic or tonic bouts within the REM epoch.
                - 'mean_duration': Mean duration of the phasic or tonic bouts in seconds.
                - 'total_duration': Total duration of the phasic or tonic periods within the REM epoch in seconds.
                - 'percent_of_rem': Percentage of the REM epoch occupied by the phasic or tonic state.

        Raises
        ------
        AttributeError
            If REM, phasic, or tonic intervals have not been detected yet.

        Examples
        --------
        >>> pt = PhasicTonic(fs=1000)
        >>> pt.detect(eeg_signal, hypnogram)
        >>> stats_df = pt.compute_stats()
        >>> print(stats_df)
        """
        stats = []

        for rem_interval in self.rem_intervals:
            # Intersect phasic and tonic intervals with the current REM interval
            phasic = rem_interval.intersect(self.phasic_intervals)
            tonic = rem_interval.intersect(self.tonic_intervals)

            for state, intervals in [("phasic", phasic), ("tonic", tonic)]:
                durations = intervals['end'] - intervals['start']
                total_duration = durations.sum()

                stats.append({
                    "rem_start": int(rem_interval["start"].item()),
                    "rem_end": int(rem_interval["end"].item()),
                    "state": state,
                    "num_bouts": len(intervals),
                    "mean_duration": durations.mean() if len(durations) > 0 else 0,
                    "total_duration": total_duration,
                    "percent_of_rem": (total_duration / rem_interval.tot_length()) * 100
                })

        return pd.DataFrame(stats)

    def get_intermediate_values(self) -> dict:
        """
        Returns thresholds and intermediate values as a dictionary.

        Returns
        -------
        dict
            Dictionary containing:

                - 'thresh_10': 10th percentile of smoothed trough differences across all epochs.
                - 'thresh_5': 5th percentile of smoothed trough differences across all epochs..
                - 'mean_inst_amp': Mean instantaneous amplitude across all REM epochs.
                - 'epoch_trough_idx': Dictionary mapping REM epoch indices to their trough indices.
                - 'epoch_smooth_diffs': Dictionary mapping REM epoch indices to their smoothed trough differences.
                - 'epoch_amplitudes': Dictionary mapping REM epoch indices to their instantaneous amplitudes.
        """
        # Check if intermediate values are available
        if self.thresholds is None:
            raise ValueError("No intermediate values available.")

        return {
            'thresh_10': self.thresholds[0],
            'thresh_5': self.thresholds[1],
            'mean_inst_amp': self.thresholds[2],
            'epoch_trough_idx': self.epoch_trough_idx,
            'epoch_smooth_diffs': self.epoch_smooth_diffs,
            'epoch_amplitudes': self.epoch_amplitudes
        }
