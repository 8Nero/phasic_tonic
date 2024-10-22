# -*- coding: utf-8 -*-
"""
Using PhasicTonic class
=========================
This tutorial covers how `PhasicTonic` is used for analysing phasic and tonic substates of REM sleep.
"""
# %%
# *** 
# Importing libraries
# -------------------
import pooch

import numpy as np
import seaborn as sns

from phasic_tonic.analysis import PhasicTonic

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
# Initialize a `PhasicTonic` instance
# -------------------------------------
# The `detect` method uses same detection algorithm as `detect_phasic`,
# with the addition of detecting tonic states. 
# The tonic states are classified as non-phasic intervals in the REM sleep.
pt = PhasicTonic(fs=fs)
results = pt.detect(eeg=lfp, hypno=hypnogram)
# %%
# The returned dictionary contains phasic and tonic intervals as
# [IntervalSet](https://pynapple.org/reference/core/interval_set/) objects from Pynapple.
print(results.keys())
print(results['phasic_intervals'])
# %%
# The `compute_stats` method can be used for computing statistics of phasic and tonic states.
stats = pt.compute_stats()
print(stats)
# %%
# 