# -*- coding: utf-8 -*-
"""
Using PhasicTonic class
=========================

`PhasicTonic` provides methods on detecting phasic and tonic states and computing simple statistical results.
This tutorial covers how it is used.
"""

# %% Importing libraries
import os
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
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
fs = 500 # Sampling rate

# %%
# ***
# Initialize a `PhasicTonic` instance
# -------------------------------------
# The `PhasicTonic` class uses same detection algorithm as `detect_phasic`. It classifies non phasic REM intervals
# as tonic REM states.
pt = PhasicTonic(fs=fs, thr_dur=900)  # thr_dur is the threshold duration of phasic REM epoch in milliseconds
results = pt.detect(eeg=lfp, hypno=hypnogram)
# %%
# The returned dictionary contains: both phasic and tonic intervals as [IntervalSet](https://pynapple.org/reference/core/interval_set/) objects from Pynapple.
print(results.keys())
print(results['phasic_intervals'])

# %%
# `compute_stats` method can be used for computing statistics of phasic and tonic states.
stats = pt.compute_stats()
print(stats)
