import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yasa
import psutil
from psutil._common import bytes2human

from pathlib import Path
from scipy.io import loadmat
from mne.filter import resample
from itertools import chain
from tqdm import tqdm
from sys import getsizeof

from func import get_sequences, phasic_detect, get_segments, create_name_cbd, create_name_rgs, create_name_os
from runtime_logger import logger_setup

import gc

CBD_DIR = "/home/nero/datasets/CBD/"
RGS_DIR = "/home/nero/datasets/RGS14/"
OS_DIR = "/home/nero/datasets/OSbasic/"

OUTPUT_DIR  = "/home/nero/datasets/preprocesssed/"
CBD_OVERVIEW_PATH = "overview.csv"

overview_df = pd.read_csv(CBD_OVERVIEW_PATH)

fs_cbd = 2500
fs_os = 2500
fs_rgs = 1000

targetFs = 500
n_down_cbd = fs_cbd/targetFs
n_down_rgs = fs_rgs/targetFs
n_down_os = fs_os/targetFs

min_dur = 2

def run_analysis(mapped):
    with tqdm(mapped.keys()) as t:
        for state in t:
            hpc = mapped[state]

            title = create_name_cbd(hpc, overview_df)
            t.set_postfix_str(title) # Set the title for the progress bar

            metadata = {}
            metaname_list  = title.split('_')
            metadata["rat_id"]    = int(metaname_list[0][3:])
            metadata["study_day"] = int(metaname_list[1][2:])
            metadata["condition"] = metaname_list[2]
            metadata["treatment"] = int(metaname_list[3])
            metadata["trial_num"] = int(metaname_list[4][-1])

            logger.debug("Loading: {0}".format(title))
            logger.debug("fname: {0}".format(state))

            # Load the LFP data
            lfpHPC = loadmat(hpc)['HPC']
            lfpHPC = lfpHPC.flatten()

            # Load the states
            hypno = loadmat(state)['states']
            hypno = hypno.flatten()

            # Skip if no REM epoch is detected
            if(not (np.any(hypno == 5))):
                logger.debug("No REM detected. Skipping.")
                continue
            elif(metadata["trial_num"] != 5):
                logger.debug("Not Trial 5.")
                continue

            logger.debug("LFP shape: {0}, size: {1}MB)".format(str(lfpHPC.shape), getsizeof(lfpHPC)//1024**2))
            logger.debug("Memory usage: {0} ({1}%)".format(bytes2human(psutil.virtual_memory().available), str(psutil.virtual_memory().percent)))

            logger.debug("STARTED: Resampling to 500 Hz.")
            # Downsample to 500 Hz
            data_resample = resample(lfpHPC, down=n_down_cbd, method='fft', verbose="INFO")
            logger.debug("FINISHED: Resampling to 500 Hz.")
            logger.debug("Resampled: {0} -> {1}.".format(str(lfpHPC.shape), str(data_resample.shape)))
            logger.debug("Memory usage: {0} ({1}%)".format(bytes2human(psutil.virtual_memory().available), str(psutil.virtual_memory().percent)))
            del lfpHPC # Save memory

            logger.debug("STARTED: Remove artifacts.")
            # Remove artifacts
            art_std, _ = yasa.art_detect(data_resample, targetFs , window=1, method='std', threshold=4, verbose='info')
            art_up = yasa.hypno_upsample_to_data(art_std, 1, data_resample, targetFs)
            data_resample[art_up] = 0
            logger.debug("FINISHED: Remove artifacts.")

            del art_std # Save memory
            del art_up # Save memory

            data_resample = data_resample - data_resample.mean()
            logger.debug("Resampled data shape: {0}, size: {1}MB)".format(str(data_resample.shape), getsizeof(data_resample)//1024**2))

            rem_seq = get_sequences(np.where(hypno == 5)[0])
            # Another representation: matrix of 2 columns and n rows (n number of rem epochs), first row is the start and second is for end idx.

            del hypno # Save memory

            # minimum duration > 2s
            logger.debug("STARTED: Extract REM epochs.")
            rem_seq = [(start, end) for start, end in rem_seq if (end-start) > min_dur]
            logger.debug("REM indices: {0}.".format(rem_seq))

            # get REM segments
            rem_idx = [(start * targetFs, (end+1) * targetFs) for start, end in rem_seq]
            rem_epochs = get_segments(rem_idx, data_resample)
            logger.debug("FINISHED: Extract REM epochs.")

            logger.debug("Memory usage: {0} ({1}%)".format(bytes2human(psutil.virtual_memory().available), str(psutil.virtual_memory().percent)))
            del data_resample # Save memory

            # Combine the REM indices with the corresponding downsampled segments
            rem = {seq:seg for seq, seg in zip(rem_seq, rem_epochs)}

            fname = OUTPUT_DIR + title
            logger.debug("Saving as: {0}.".format(title))
            np.save(fname, rem)
            
            gc.collect() # Force garbage collection
            

if __name__ == "__main__":
    logger = logger_setup()
    logger.debug("Saving to: {0}".format(OUTPUT_DIR))

    pattern = r"[\w-]+posttrial[\w-]+"
    mapped = {}

    for root, dirs, fils in os.walk(CBD_DIR):
        for dir in dirs:
            # Check if the directory is a post trial directory
            if re.match(pattern, dir, flags=re.IGNORECASE):
                dir = Path(os.path.join(root, dir))
                HPC_file = next(dir.glob("*HPC*continuous*"))
                states = next(dir.glob('*-states*'))
                mapped[str(states)] = str(HPC_file)

    print("Number of recordings: ", len(mapped))
    run_analysis(mapped)

    