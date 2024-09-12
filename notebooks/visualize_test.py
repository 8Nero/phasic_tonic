# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 21:50:34 2024

@author: animu
"""
import sys
if "C:/Users/animu/phasic_tonic/src" not in sys.path:
    sys.path.append("C:/Users/animu/phasic_tonic/src")

from phasic_tonic.utils import preprocess
from scipy.io import loadmat

from phasic_tonic.detect_phasic import compute_thresholds, get_rem_epochs, get_phasic_candidates, is_valid_phasic
from phasic_tonic.utils import get_start_end

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import spectrogram
import pynapple as nap

class PhasicTonic():
    def __init__(self, fs, thr_dur):
        self.fs = fs
        self.thr_dur = thr_dur
        self.eeg = None
        self.t = None
        self.hypno = None
        self.rem_interval = None
        self.phasic_interval = None
        self.tonic_interval = None
        self.thresholds = None
        self.trough_idx_seq = None
        self.smooth_difference_seq = None
        self.eeg_seq = None
        
    def fit(self, eeg, hypno):
        self.t = np.arange(0, len(eeg)/self.fs, 1/self.fs)
        self.eeg = nap.Tsd(t=self.t, d=eeg)
        self.hypno= hypno
        
        rem_start, rem_end = get_start_end(hypno, 5)
        self.rem_interval = nap.IntervalSet(rem_start, rem_end)
        
        rem_epochs = get_rem_epochs(self.eeg.to_numpy(), hypno, self.fs)
        thresholds, trough_idx_seq, smooth_difference_seq, eeg_seq = compute_thresholds(rem_epochs, self.fs)
        self.thresholds = thresholds
        self.trough_idx_seq = trough_idx_seq
        self.smooth_difference_seq = smooth_difference_seq
        self.eeg_seq = eeg_seq
        thr1, thr2, thr3 = thresholds
        
        phasicREM = {rem_idx: [] for rem_idx in rem_epochs.keys()}
   
        for rem_idx, trough_idx in trough_idx_seq.items():
            rem_start, rem_end = rem_idx
            offset = rem_start * self.fs
            smooth_difference, eegh = smooth_difference_seq[rem_idx], eeg_seq[rem_idx]
           
            candidates = get_phasic_candidates(smooth_difference, trough_idx, thr1, self.thr_dur, self.fs)
           
            for start, end in candidates:
                if is_valid_phasic(start, end, smooth_difference, eegh, trough_idx, thr2, thr3):
                    t_a = trough_idx[start] + offset
                    t_b = min(trough_idx[end] + offset, rem_end * self.fs)
                    phasicREM[rem_idx].append((t_a, t_b + 1))

        # Create interval sets for phasic, tonic REM
        start, end = [], []
        for rem_idx in phasicREM:
            for s, e in phasicREM:
                start.append(s/self.fs)
                end.append(e/self.fs)
        self.phasic_interval = nap.IntervalSet(start, end)
        self.tonic_interval = self.rem_interval.set_diff(self.phasic_interval)
        return phasicREM
    
    def plot(self):
        nsr_seg = 1
        perc_overlap = 0.8
        vm = 3000
        
        # Define the custom colors
        colors = [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8]]
        
        # Create a custom colormap
        my_map = LinearSegmentedColormap.from_list('brs', colors, N=5)
        
        freq, t, SP = spectrogram(self.eeg, fs=self.fs, window='hann', 
                                  nperseg=int(nsr_seg * self.fs), 
                                  noverlap=int(nsr_seg * self.fs* perc_overlap))
        
        ifreq = np.where(freq <= 20)[0]
        
        gamma = (50, 90)
        df = freq[1] - freq[0]
        igamma = np.where((freq >= gamma[0]) & (freq <= gamma[1]))[0]
        pow_gamma = SP[igamma,:].sum(axis=0) * df
        
        fig = plt.figure(figsize=(12,6), layout='constrained')
        #fig.suptitle(name, fontsize=12)
        axs = fig.subplot_mosaic([["states"],
                          ["lfp"],
                          ["phasic"],
                          ["iti"],
                          ["spectrogram"],
                          ["gamma"]], sharex=True,
                         gridspec_kw = {'height_ratios':[1, 8, 1, 8, 8, 8],
                                        'hspace':0.05}
                         )
        # Plot sleep states
        tmp = axs["states"].pcolorfast(self.t, [0, 1], np.array([self.hypno]), vmin=1, vmax=5)
        tmp.set_cmap(my_map)
        _despine_axes(axs["states"])            
        
        # Plot HPC region
        axs["lfp"].plot(self.t, self.eeg, color='k')
        
        # Plot spectrogram
        axs["spectrogram"].pcolorfast(self.t, freq[ifreq], SP[ifreq, :], vmin=0, vmax=vm, cmap='hot')
        axs["spectrogram"].set_ylabel("Freq. (Hz)")

        # Plot phasicREM as spikes
        axs["phasic"].set_ylabel("Phasic")
        axs["phasic"].eventplot((self.phasic_interval["end"]+self.phasic_interval["start"])/2)
        _despine_axes(axs["phasic"])

        
        # Plot inter-trough intervals (iti)
        for epoch in self.rem_interval:
            rem_start, rem_end = int(epoch["start"].item()), int(epoch["end"].item())
            axs["lfp"].axvspan(rem_start, rem_end, facecolor=[0.7, 0.7, 0.8], alpha=0.4)
            
            tridx = self.trough_idx_seq[(rem_start, rem_end)] 
            sdiff = self.smooth_difference_seq[(rem_start, rem_end)]
            eegh = self.eeg_seq[(rem_start, rem_end)]
            
            rem_start *= self.fs
            rem_end = (rem_end + 1)*self.fs

            tridx = (tridx + rem_start)/self.fs
            axs["iti"].plot(tridx[:-1], sdiff, drawstyle="steps-pre", color='k')
            axs["lfp"].plot(self.t[rem_start:rem_end], eegh, 'y', '--')
            axs["lfp"].plot([self.t[rem_start], self.t[rem_end]], [self.thresholds[2], self.thresholds[2]], 'r', '--')

            
        # Plot phasicREM
        [axs["lfp"].plot(self.eeg.restrict(self.phasic_interval[i]), color='r') for i in range(len(self.phasic_interval))]
        
        axs["iti"].axhline(y=self.thresholds[0], color='r', linestyle='--')
        axs["iti"].axhline(y=self.thresholds[1], color='y', linestyle='--')
        axs["iti"].set_ylabel("ITI")
        
        axs["gamma"].plot(t, pow_gamma, '.-')
        axs["gamma"].set_ylabel(r'$\gamma$')
        
        axs["lfp"].set_ylabel("LFP")
        
def _despine_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
#%% Loading
lfp = loadmat("C:/Users/animu/phasic_tonic/data/example/HPC_100_CH15.continuous_merged.mat")['HPC'].flatten()
sleep = loadmat("C:/Users/animu/phasic_tonic/data/example/post_trial5_2017-11-16_14-46-12-states.mat")['states'].flatten()

lfp = preprocess(lfp, 5)
#%% Plotting
g = PhasicTonic(fs=500, thr_dur=900)
g.fit(lfp, sleep)
g.plot()

#
        
        
        
        
        
        
