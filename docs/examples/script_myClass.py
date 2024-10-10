"""
PhasicTonic class
"""

from phasic_tonic.utils import preprocess
from phasic_tonic.PhasicTonic import PhasicTonic

import numpy as np
import pandas as pd
from scipy.io import loadmat

path = "/home/nero/datasets/OSbasic/11/Rat-OS-Ephys_Rat11_SD1_CON_29-10-2018/2018-10-29_12-41-45_Post_Trial2"
lfp = loadmat(path + "/HPC_100_CH32_0.continuous.mat")['HPC'].flatten()
sleep = loadmat(path + "/2018-10-29_12-41-45_post_trial2-states.mat")['states'].flatten()
lfp = preprocess(lfp, 5)

#%% Plotting
g = PhasicTonic(fs=500, thr_dur=900)
phrem = g.fit(lfp, sleep)
g.plot()
df = g.compute_stats()
