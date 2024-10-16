# -*- coding: utf-8 -*-
"""
Detecting Phasic REM states
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