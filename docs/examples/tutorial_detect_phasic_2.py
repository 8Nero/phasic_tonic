"""
Understanding the detect_phasic algorithm
===========================================

The `detect_phasic` function is a threshold based algorithm for identifying phasic REM states within Local Field Potential (LFP) data.
This tutorial covers the implementation of [Mizuseki et al. (2011)](https://doi.org/10.1038/nn.2894)
"""
# %% 
# Importing libraries
import pooch

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# mkdocs_gallery_thumbnail_number = 6

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(context='notebook', style="white", rc=custom_params)
# %% 
# ***
# Preparing the data
# ------------------
# We download an example data.
f_path = pooch.retrieve("https://raw.githubusercontent.com/8Nero/phasic_tonic/main/data/ex01.npz", 
                        known_hash="11e579b9305859db9101ba3e227e2d9a72008423b3ffe2ad9a8fee1765236663")
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
# Selecting REM epochs
# --------------------
# We extract REM epochs from the signal.
from phasic_tonic.core import get_rem_epochs

rem_epochs = get_rem_epochs(lfp, hypnogram, fs)
print(rem_epochs)
# %%
# ***
# Bandpass filtering in the Theta region
# --------------------------------------
# We will isolate the theta frequency band (5-12 Hz) from the LFP.
# Phasic REM states are characterized by transient accelerations in theta rhythm, typically lasting around 2 seconds.
# During these events, there's an increase in high-frequency oscillations and enhanced coherence within hippocampal circuits, particularly between theta and gamma bands.
# This enhanced activity is believed to facilitate memory consolidation by boosting communication between the hippocampus and neocortical regions, such as the retrosplenial cortex, allowing for effective information transfer during these brief windows of heightened connectivity. [Gomes de Almeida-Filho et al. (2021)](https://doi.org/10.1038/s41598-021-91659-5)
from neurodsp.filt import filter_signal

epoch = list(rem_epochs.values())[1]
epoch_filt = filter_signal(epoch, fs, 'bandpass', (5, 12), remove_edges=False)
# %%
# We pick a REM epoch and plot the filtered signal in the interval of 5 seconds.
time = np.arange(0, len(epoch)/fs, 1/fs)

fig, ax = plt.subplots(1, constrained_layout=True, figsize=(10, 3))
ax.plot(time[:5*fs], epoch[:5*fs])
ax.plot(time[:5*fs], epoch_filt[:5*fs])
ax.set_xlabel('Time (s)')
ax.set_ylabel('LFP')
ax.legend(['Original', 'Filtered'])
# %%
# ***
# Hilbert Transform for Instantaneous Amplitude and Phase
# -------------------------------------------------------
# We apply Hilbert transform to the theta oscillation to obtain its analytic signal representation.
# We then compute the instantaneous amplitude (envelope) and instantaneous phase from the analytic signal.
from scipy.signal import hilbert

analytic_signal = hilbert(epoch_filt)
inst_phase, inst_amp = np.angle(analytic_signal), np.abs(analytic_signal)
#%%#
fig, axes = plt.subplots(2, 1, constrained_layout=True, figsize=(10, 6), sharex=True)

axes[0].plot(time[:5*fs], epoch_filt[:5*fs], label='Original signal')
axes[0].plot(time[:5*fs], inst_amp[:5*fs], label='Instantaneous Amplitude')
axes[0].set_ylabel('LFP')
axes[0].set_xlabel('Time (s)')
axes[0].legend()

axes[1].plot(time[:5*fs], inst_phase[:5*fs])
axes[1].set_ylabel('Instantaneous Phase')
axes[1].set_xlabel('Time (s)')
# %%
# ***
# Inter-Trough intervals
# ----------------------
# We detect troughs based on the instantaneous phase.
from phasic_tonic.core import detect_troughs

troughs = detect_troughs(inst_phase)

fig, ax = plt.subplots(1, constrained_layout=True, figsize=(10, 3))

ax.plot(time, epoch_filt)
ax.scatter(time[troughs], epoch_filt[troughs], c='r', label='Troughs')

ax.set_xlim((0,5))
ax.set_xlabel('Time (s)')
ax.set_ylabel('LFP')
ax.legend()
# %%
# By computing trough differenes and applying smooth filter. We get inter-trough intervals.
from phasic_tonic.core import smooth_signal

smooth_diffs = smooth_signal(np.diff(troughs))
fig, axes = plt.subplots(2, 1, constrained_layout=True, figsize=(10, 6), sharex=True)

axes[0].plot(time, epoch_filt)
axes[0].scatter(time[troughs], epoch_filt[troughs], c='r', label='Troughs')

axes[0].set_ylabel('LFP')
axes[0].legend()

axes[1].plot(time[troughs[:-1]], smooth_diffs, drawstyle="steps-pre", color='k')
axes[1].set_ylabel("Inter-Trough interval (s)")
axes[1].set_xlabel('Time (s)')
axes[1].set_xlim((0,5))
# %%
# ***
# Threshold comparison
# --------------------
# We will use `compute_thresholds` function to compute 10th and 5th percentile of inter-trough intervals
# and the mean instantaneous amplitude during the entire REM sleep.
from phasic_tonic.core import compute_thresholds

thresholds, _, _, _ = compute_thresholds(rem_epochs, fs)
thresh_10, thresh_5, mean_amp = thresholds
# %%
# We plot the thresholds within our chosen interval.
# ----------------- ---------------------------------
# We see that the segment in the beginning has inter-trough intervals less than Threshold 1 which
# makes it a candidate epoch for phasic state. We see that it's also below Threshold 2 and its
# mean instantaneous amplitude seems to greater than the threshold.
 
fig, axes = plt.subplots(2, 1, constrained_layout=True, figsize=(10, 6), sharex=True)

axes[0].plot(time, epoch_filt, label='Original signal')
axes[0].plot(time, inst_amp, label='Instantaneous Amplitude')
axes[0].axhline(y=mean_amp, color='r', linestyle='--', label="Mean Instantaneous Amplitude")
axes[0].set_ylabel('LFP')
axes[0].set_xlabel('Time (s)')
axes[0].set_xlim((0,5))
axes[0].legend()

axes[1].plot(time[troughs[:-1]], smooth_diffs, drawstyle="steps-pre", color='k')
axes[1].set_ylabel("Inter-Trough interval (ms)")
axes[1].set_xlabel('Time (s)')

axes[1].axhline(y=thresh_5, color='r', linestyle='--', label='Threshold 2')
axes[1].axhline(y=thresh_10, color='y', linestyle='--', label="Threshold 1")
axes[1].legend()

# %%
# We will verify that it does indeed fulfill the criteria. 
from phasic_tonic.core import get_phasic_candidates

candidates = get_phasic_candidates(smooth_diffs, troughs, thresh_10, thr_dur=900, fs=fs)
print(candidates)
# %%
# In general, candidate epochs are region where inter-trough intervals are less than
# the 10th percentile. The candidate epochs are considered phasic REM epochs if following criteria is fulfullied:
###############################################################################
# 1. The duration of an epoch is longer than 900 ms.
###############################################################################
# 2. The minimum of inter-trough intervals during the epoch is less than 5th percentile of entire inter-trough intervals.
###############################################################################
# 3. The mean amplitude during an epoch was greater than the mean amplitude during the entire REM sleep. 

# %%
from phasic_tonic.core import is_valid_phasic

valid_periods = []
for start, end in candidates:
    smoothed_diffs_slice = smooth_diffs[start:end]
    inst_amp_slice = inst_amp[troughs[start]:troughs[end] + 1]

    if is_valid_phasic(smoothed_diffs_slice, inst_amp_slice, thresh_10, mean_amp):
        start_time = troughs[start]
        end_time = troughs[end]
        valid_periods.append((int(start_time), int(end_time) + 1))

print(valid_periods)
# %%
# We see that it is indeed phasic REM epoch.
fig, axes = plt.subplots(2, 1, constrained_layout=True, figsize=(10, 6), sharex=True)

axes[0].plot(time, epoch_filt)
axes[0].plot(time, inst_amp)
axes[0].axhline(y=mean_amp, color='r', linestyle='--')
axes[0].set_ylabel('LFP')
axes[0].set_xlabel('Time (s)')
axes[0].set_xlim((0,5))

axes[1].plot(time[troughs[:-1]], smooth_diffs, drawstyle="steps-pre", color='k')
axes[1].set_ylabel("Inter-Trough interval (ms)")
axes[1].set_xlabel('Time (s)')

axes[1].axhline(y=thresh_5, color='r', linestyle='--')
axes[1].axhline(y=thresh_10, color='y', linestyle='--')
for start, end in valid_periods:
    axes[0].axvspan(start/fs, end/fs, alpha=0.2, color='r')
    axes[1].axvspan(start/fs, end/fs, alpha=0.2, color='r')
# %%
# We can also verify that
period = valid_periods[0]
print(inst_amp[troughs[period[0]]:troughs[period[1]]+1].mean() >= mean_amp)
print(smooth_diffs[period[0]:period[1]].min() < thresh_5)
