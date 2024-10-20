# -*- coding: utf-8 -*-
"""
Using the detect_phasic function
==================================

The `detect_phasic` function detects phasic REM periods in EEG data based on the method described by [Mizuseki et al. (2011)](https://doi.org/10.1038/nn.2894).
This tutorial covers the use of `detect_phasic` function.
"""
# %% Importing libraries
import os
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from phasic_tonic.detect import detect_phasic
# mkdocs_gallery_thumbnail_number = 4

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(context='notebook', style="white", rc=custom_params)
# %% 
# ***
# Preparing the data
# ------------------
# We download an example data.
file = 'ex01.npz'
if file not in os.listdir("."):
    urlretrieve("https://raw.githubusercontent.com/8Nero/phasic_tonic/main/data/ex01.npz", file)
# %%
# ***
# Loading the data
# ----------------
data = np.load(file, allow_pickle=True)

hypnogram = data['hypnogram']
lfp = data['lfp_hpc']
fs = 500  # Sampling rate
# %%
# ***
# Plotting the hypnogram
# ----------------------
time = np.arange(len(hypnogram))

fig, ax = plt.subplots(1, constrained_layout=True, figsize=(10, 3))
ax.step(time, hypnogram)
ax.set_yticks([1, 3, 5], ["Wake", "NREM", "REM"])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Stage')
# %% 
# Plotting the Local Field Potential data
# ---------------------------------------
# We plot in the interval of 5 seconds during REM sleep.
time = np.arange(0, len(lfp)/fs, 1/fs)

fig, ax = plt.subplots(1, constrained_layout=True, figsize=(10, 3))
ax.plot(time[1600*fs:1605*fs], lfp[1600*fs:1605*fs])
ax.set_xlabel('Time (s)')
ax.set_ylabel('LFP')

# %% 
# ***
# Calling the `detect_phasic` function
# ------------------------------------
phasicREM = detect_phasic(lfp, hypnogram, fs)
# %%
# ***
# Analysing the output
# --------------------
# The function returns a dictionary where each key is a tuple of REM epoch timestamps in seconds, and the 
# corresponding value is a list of phasic REM epochs in sampling points.
for rem_timestamp, phasic_epochs in phasicREM.items():
    print(f"REM Epoch: {rem_timestamp}, detected phasic REM epochs: {phasic_epochs}")
# %%
# ***
# Plotting phasic REM events
# --------------------------
time = np.arange(0, len(lfp)/fs, 1/fs)

fig, ax = plt.subplots(1, constrained_layout=True, figsize=(10, 3))
ax.plot(time, lfp)
ax.set_xlabel('Time (s)')
ax.set_ylabel('LFP')

# Mark phasic states on the plot
for rem_timestamp, phasic_epochs in phasicREM.items():
    ax.axvspan(rem_timestamp[0], rem_timestamp[1], color='black', alpha=0.2)  
    for event in phasic_epochs:
      ax.axvspan(event[0]/fs, event[1]/fs, color='red', alpha=0.5)

# %%
# Create a grid of subplots for REM episodes
# -------------
fig, axes = plt.subplots(2, 2, constrained_layout=True, figsize=(12, 8))
axes = axes.flatten()

for i, (rem_timestamp, phasic_epochs) in enumerate(phasicREM.items()):
    ax = axes[i]
    
    start_time, end_time = rem_timestamp
    time_window = (time >= start_time) & (time <= end_time)
    
    ax.plot(time[time_window], lfp[time_window])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('LFP')
    ax.set_title(f'REM Epoch {i+1}')
    ax.set_ylim(-500, 500)
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    
    for event in phasic_epochs:
        ax.axvspan(event[0] / fs, event[1] / fs, color='red', alpha=0.5)

    ax.set_xlim(start_time, end_time)
