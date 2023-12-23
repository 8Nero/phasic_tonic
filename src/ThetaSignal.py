import numpy as np
from bycycle.features import compute_features
from src.Signal import Signal
from src.pipeline import get_periods

class ThetaSignal(Signal):
    def __init__(self, array: np.ndarray, sampling_rate: int):
        super().__init__(array, sampling_rate)
        
        self.phasic = []
        self.tonic = []
    
    def segment_cycles(self, f_range: any, skip_threshold : int, threshold_kwargs : dict):
        """
        Segment the signal into phasic and tonic components using the Cycle-by-Cycle algorithm.
        
        Parameters:
            f_range (tuple of (float, float)): Frequency range for narrowband signal of interest (Hz).
            skip_threshold (int): Threshold value for connecting two consecutive segments.
            threshold_kwargs (dict): Parameters for the Cycle-by-Cycle algorithm

        """

        # Run Cycle by Cycle algorithm for burst detection
        df = compute_features(self.filtered, 
                              self.sampling_rate, 
                              f_range=f_range, 
                              threshold_kwargs=threshold_kwargs, 
                              center_extrema='peak')
        
        df = df[["sample_last_trough", "sample_next_trough", "is_burst"]]
        phasic_df = df[df["is_burst"] == True]
        tonic_df = df[df["is_burst"] == False]

        result_msg = "{} periods in the {} signal: {}"
        print(result_msg.format("Phasic", self.filter_type, len(phasic_df)))
        print(result_msg.format("Tonic", self.filter_type, len(tonic_df)))

        if len(phasic_df) != 0:
            self.phasic = get_periods(phasic_df[["sample_last_trough", "sample_next_trough"]],
                                            skip_threshold=skip_threshold)
        if len(tonic_df) != 0:
            self.tonic = get_periods(tonic_df[["sample_last_trough", "sample_next_trough"]],
                                            skip_threshold=skip_threshold)

    def get_phasic(self):
        segments = []
        for period in self.phasic:
            start, end = period
            segments.append(self.filtered[start:end])
        return segments
    
    def get_tonic(self):
        segments = []
        for period in self.tonic:
            start, end = period
            segments.append(self.filtered[start:end])
        return segments