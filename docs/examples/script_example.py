# -*- coding: utf-8 -*-
"""
# Load the datasets

The path to dataset directory and patterns to search in those directories for the HPC, PFC recordings are loaded from the config file.
"""

from phasic_tonic.detect_phasic import detect_phasic
from phasic_tonic.DatasetLoader import DatasetLoader
from phasic_tonic.helper import get_metadata
from phasic_tonic.runtime_logger import logger_setup    
from phasic_tonic.utils import get_sequences, get_start_end, preprocess

import numpy as np
import pandas as pd
import pynapple as nap

from tqdm.auto import tqdm
from scipy.io import loadmat

fs_cbd = 2500
fs_os = 2500
fs_rgs = 1000

targetFs = 500
n_down_cbd = fs_cbd/targetFs
n_down_rgs = fs_rgs/targetFs
n_down_os = fs_os/targetFs

logger = logger_setup()

CONFIG_DIR = "/home/nero/phasic_tonic/data/dataset_loading.yaml"

Datasets = DatasetLoader(CONFIG_DIR)
mapped_datasets = Datasets.load_datasets()
#%% Check the number of recordings
cbd_cnt = 0
rgs_cnt = 0
os_cnt = 0

# Count recordings belonging to CBD dataset
for name in mapped_datasets:
    metadata = get_metadata(name)
    if metadata['treatment'] == 0 or metadata['treatment'] == 1:
        cbd_cnt += 1
    elif metadata['treatment'] == 2 or metadata['treatment'] == 3:
        rgs_cnt += 1
    elif metadata['treatment'] == 4:
        os_cnt += 1

assert cbd_cnt == 170
assert rgs_cnt == 159
assert os_cnt == 210
#%% Loop through the datasets
with tqdm(mapped_datasets) as mapped_tqdm:
    for name in mapped_tqdm:
        metadata = get_metadata(name)
        mapped_tqdm.set_postfix_str(name)
        states_fname, hpc_fname, pfc_fname = mapped_datasets[name]
        logger.debug("Loading: {0}".format(name))

        if metadata["treatment"] == 0 or metadata["treatment"] == 1:
            n_down = n_down_cbd
        elif metadata["treatment"] == 2 or metadata["treatment"] == 3:
            n_down = n_down_rgs
        elif metadata["treatment"] == 4:
            n_down = n_down_os
        
        # Load the LFP data
        lfpHPC = loadmat(hpc_fname)['HPC'].flatten()
        lfpPFC = loadmat(pfc_fname)['PFC'].flatten()

        # Load the states
        hypno = loadmat(states_fname)['states'].flatten()
        
        # Skip if no REM epoch is detected
        if(not (np.any(hypno == 5))):
            logger.debug("No REM detected. Skipping.")
            continue
        elif(np.sum(np.diff(get_sequences(np.where(hypno == 5)[0]))) < 10):
            logger.debug("No REM longer than 10s. Skipping.")
            continue
        
        # Create Pynapple IntervalSet        
        start, end = get_start_end(hypno, sleep_state_id=5)
        rem_interval = nap.IntervalSet(start=start, end=end)
        
        # Create TsdFrame for HPC and PFC signals
        fs = n_down*targetFs
        t = np.arange(0, len(lfpHPC)/fs, 1/fs)
        lfp = nap.TsdFrame(t=t, d=np.vstack([lfpHPC, lfpPFC]).T, columns=['HPC', 'PFC'])
        
        # Detect phasic intervals
        lfpHPC_down = preprocess(lfpHPC, n_down)
        phREM = detect_phasic(lfpHPC_down, hypno, targetFs)
        
        # Create phasic REM IntervalSet
        start, end = [], []
        for rem_idx in phREM:
            for s, e in phREM[rem_idx]:
                start.append(s/targetFs)
                end.append(e/targetFs)
        phasic_interval = nap.IntervalSet(start, end)
        tonic_interval = rem_interval.set_diff(phasic_interval)
#%% Access the HPC and PFC signals during phasic REM 
lfp.restrict(phasic_interval)