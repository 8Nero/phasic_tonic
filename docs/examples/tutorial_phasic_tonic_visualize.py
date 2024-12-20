# -*- coding: utf-8 -*-
"""
Visualizing phasic and tonic REM states
=======================================

This tutorial covers an example of how `PhasicTonic` detector instance can be used to create
visualizations of phasic and tonic REM detections.
"""
# %% 
# Importing libraries
import pooch

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import spectrogram

from phasic_tonic.analysis import PhasicTonic

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(context='notebook', style="white", rc=custom_params)
# %%
# ***
# Preparing the data
# ------------------
# We download an example data.
f_path = pooch.retrieve("https://raw.githubusercontent.com/8Nero/phasic_tonic/main/data/ex02.npz",
                        known_hash="382aa2be86f73ebbf24a9b759a47e35d1d751e161c280ff027bcf9f1068de674")
# %%
# ***
# Loading the data
# ----------------
data = np.load(f_path, allow_pickle=True)

hypnogram = data['hypnogram']
lfp = data['lfp_hpc']
fs = 500  # Sampling rate

# %%
# ***
# Detection
# --------------------

# Instantiate the PhasicTonic class
pt = PhasicTonic(fs=fs)

# Detect phasic and tonic REM states
pt.detect(lfp, hypnogram)

# %%
# ***
# Access intermediate values
# --------------------
# We can access the thresholds and intermediate values used for detection from the attributes.
rem_intervals = pt.rem_intervals
phasic_intervals = pt.phasic_intervals
thresholds = pt.thresholds
epoch_trough_idx = pt.epoch_trough_idx
epoch_smooth_diffs = pt.epoch_smooth_diffs
# %%
# ***
# Creating the visualization
# --------------------------

fig = plt.figure(figsize=(12, 6), layout='constrained')
axs = fig.subplot_mosaic([
    ["states"],
    ["lfp"],
    ["iti"],
    ["spectrogram"],
], sharex=True, gridspec_kw={'height_ratios': [1, 8, 8, 8], 'hspace': 0.05})
time = np.arange(0, len(lfp)/fs, 1/fs)

nsr_seg, perc_overlap, vm = 1, 0.8, 3000

# sleep states
colors = [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8]]
my_map = LinearSegmentedColormap.from_list('brs', colors, N=5)
tmp = axs['states'].pcolorfast(time, [0, 1], np.array([hypnogram]), vmin=1, vmax=5)
tmp.set_cmap(my_map)

axs['states'].spines["top"].set_visible(False)
axs['states'].spines["right"].set_visible(False)
axs['states'].spines["bottom"].set_visible(False)
axs['states'].spines["left"].set_visible(False)
axs['states'].axes.get_xaxis().set_visible(False)
axs['states'].axes.get_yaxis().set_visible(False)

# LFP
axs['lfp'].plot(time, lfp)
axs['lfp'].set_xlabel('Time (s)')
axs['lfp'].set_ylabel('LFP')

# Mark phasic states on the plot
for rem_interval in pt.rem_intervals:
    rem_start, rem_end = rem_interval['start'].item(), rem_interval['end'].item()
    axs['lfp'].axvspan(rem_start, rem_end, color='black', alpha=0.1)

[axs['lfp'].axvspan(phasic_interval['start'].item(), phasic_interval['end'].item(), color='r', alpha=0.5) for phasic_interval in pt.phasic_intervals]

# Spectrogram
freq, t, SP = spectrogram(lfp, fs=fs, window='hann', 
                          nperseg=int(nsr_seg * fs), 
                          noverlap=int(nsr_seg * fs * perc_overlap))
ifreq = np.where(freq <= 20)[0]
axs['spectrogram'].pcolorfast(t, freq[ifreq], SP[ifreq, :], vmin=0, vmax=vm, cmap='hot')
axs['spectrogram'].set_ylabel("Freq. (Hz)")

# Inter-trough intervals
for epoch in pt.rem_intervals:
    rem_start, rem_end = int(epoch["start"].item()), int(epoch["end"].item())
    tridx = pt.epoch_trough_idx[(rem_start, rem_end)]
    sdiff = pt.epoch_smooth_diffs[(rem_start, rem_end)]
    tridx = (tridx + rem_start * fs) / fs
    axs["iti"].plot(tridx[:-1], sdiff, drawstyle="steps-pre", color='k')
axs["iti"].axhline(y=pt.thresholds[0], color='r', linestyle='--')
axs["iti"].axhline(y=pt.thresholds[1], color='y', linestyle='--')
axs["iti"].set_ylabel("ITI")
# %%
# 