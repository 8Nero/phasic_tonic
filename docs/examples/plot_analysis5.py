# -*- coding: utf-8 -*-
"""
Phasic and Tonic states analysis
"""
# Importing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from scipy.io import loadmat

from phasic_tonic.analysis import PhasicTonic

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
seaborn.set_theme(context='notebook', style="ticks", rc=custom_params)

# Load sample data
path = "/home/nero/datasets/OSbasic/11/Rat-OS-Ephys_Rat11_SD1_CON_29-10-2018/2018-10-29_12-41-45_Post_Trial2"
lfp = loadmat(path + "/HPC_100_CH32_0.continuous.mat")['HPC'].flatten()
hypnogram = loadmat(path + "/2018-10-29_12-41-45_post_trial2-states.mat")['states'].flatten()
fs=2500
#lfp = preprocess(lfp, 5)

#%%
# ***
# Initialize the `PhasicTonic` Detector
# -------------------------------------
# The `PhasicTonic` class performs the detection based on the method described by Mizuseki et al. (2011).
pt_detector = PhasicTonic(fs=fs, thr_dur=900)  # thr_dur is the threshold duration in milliseconds
results = pt_detector.detect(eeg=lfp, hypno=hypnogram)
#%%
# This method will process the EEG data and hypnogram to detect phasic and tonic REM periods. The `results` dictionary contains:
# - `phasic_intervals`: IntervalSet of phasic REM periods
# - `tonic_intervals`: IntervalSet of tonic REM periods
print(results)

#%% Access and Plot Intermediate Values
#Intermediate values can provide valuable insights into the detection process. You can access them using:
intermediate_values = pt_detector.get_intermediate_values()
#%% Plotting Example: Smoothed Trough Differences and Thresholds

# Extract values
epoch_smooth_diffs = intermediate_values['epoch_smooth_diffs']
thresh_10 = intermediate_values['thresh_10']
thresh_5 = intermediate_values['thresh_5']

#%% Plot smoothed trough differences for each REM epoch
for rem_idx, smooth_diffs in epoch_smooth_diffs.items():
    plt.figure(figsize=(12, 6))
    plt.plot(smooth_diffs, label='Smoothed Trough Differences')
    plt.axhline(y=thresh_10, color='r', linestyle='--', label='10th Percentile Threshold')
    plt.axhline(y=thresh_5, color='g', linestyle='--', label='5th Percentile Threshold')
    plt.title(f'REM Epoch {rem_idx}')
    plt.xlabel('Trough Index')
    plt.ylabel('Smoothed Difference')
    plt.legend()
    plt.show()

#%% Plotting Example: Instantaneous Amplitudes

# Extract values
epoch_amplitudes = intermediate_values['epoch_amplitudes']
mean_inst_amp = intermediate_values['mean_inst_amp']

# Plot instantaneous amplitudes for each REM epoch
for rem_idx, inst_amp in epoch_amplitudes.items():
    plt.figure(figsize=(12, 6))
    plt.plot(inst_amp, label='Instantaneous Amplitude')
    plt.axhline(y=mean_inst_amp, color='r', linestyle='--', label='Mean Instantaneous Amplitude')
    plt.title(f'REM Epoch {rem_idx}')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

#%% Analyze Detection Results
stats_df = pt_detector.compute_stats()
print(stats_df)
#%% Plot EEG signal with phasic and tonic intervals highlighted
plt.figure(figsize=(15, 6))
time_axis = np.arange(len(lfp)) / fs
plt.plot(time_axis, lfp, label='EEG Signal', alpha=0.5)

phasic_intervals = results['phasic_intervals']
tonic_intervals = results['tonic_intervals']

# Highlight phasic intervals
for interval in phasic_intervals:
    plt.axvspan(interval['start'].item(), interval['end'].item(), color='red', alpha=0.3, label='Phasic REM')

# Highlight tonic intervals
for interval in tonic_intervals:
    plt.axvspan(interval['start'].item(), interval['end'].item(), color='blue', alpha=0.3, label='Tonic REM')

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('EEG Signal with Phasic and Tonic REM Intervals')

plt.legend(handles=[
    plt.Line2D([0], [0], color='red', lw=4, alpha=0.3, label='Phasic REM'),
    plt.Line2D([0], [0], color='blue', lw=4, alpha=0.3, label='Tonic REM')
])
plt.show()
