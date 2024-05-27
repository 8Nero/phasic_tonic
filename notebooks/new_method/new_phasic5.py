import os
import re
import numpy as np
import pandas as pd
import yasa
import psutil
from psutil._common import bytes2human

from pathlib import Path
from scipy.io import loadmat
from mne.filter import resample
from tqdm import tqdm
from sys import getsizeof

from func import get_sequences, get_segments, create_name_cbd, create_name_rgs, create_name_os
from runtime_logger import logger_setup

import gc

CBD_DIR = "/home/nero/datasets/CBD/"
RGS_DIR = "/home/nero/datasets/RGS14/"
OS_DIR = "/home/nero/datasets/OSbasic/"

OUTPUT_DIR  = "/home/nero/datasets/preprocessed/"
CBD_OVERVIEW_PATH = "/home/nero/phasic_tonic/notebooks/new_method/overview.csv"

fs_cbd = 2500
fs_os = 2500
fs_rgs = 1000

targetFs = 500
n_down_cbd = fs_cbd/targetFs
n_down_rgs = fs_rgs/targetFs
n_down_os = fs_os/targetFs

min_dur = 2

def run(mapped, name_func, n_down):
  with tqdm(mapped.keys()) as t:
      for state in t:
          hpc = mapped[state]
  
          title = name_func(hpc)
          t.set_postfix_str(title) # Set the title for the progress bar
  
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

          logger.debug("LFP shape: {0}, size: {1}MB)".format(str(lfpHPC.shape), getsizeof(lfpHPC)//1024**2))
          logger.debug("Memory usage: {0} ({1}%)".format(bytes2human(psutil.virtual_memory().available), str(psutil.virtual_memory().percent)))
  
          logger.debug("STARTED: Resampling to 500 Hz.")
          # Downsample to 500 Hz
          data_resample = resample(lfpHPC, down=n_down, method='fft', npad='auto')
          logger.debug("FINISHED: Resampling to 500 Hz.")
          logger.debug("Resampled: {0} -> {1}.".format(str(lfpHPC.shape), str(data_resample.shape)))
          logger.debug("Memory usage: {0} ({1}%)".format(bytes2human(psutil.virtual_memory().available), str(psutil.virtual_memory().percent)))
  
          logger.debug("STARTED: Remove artifacts.")
          # Remove artifacts
          art_std, _ = yasa.art_detect(data_resample, targetFs , window=1, method='std', threshold=4, verbose='info')
          art_up = yasa.hypno_upsample_to_data(art_std, 1, data_resample, targetFs)
          data_resample[art_up] = 0
          logger.debug("FINISHED: Remove artifacts.")
  
          data_resample = data_resample - data_resample.mean()
          logger.debug("Resampled data shape: {0}, size: {1}MB)".format(str(data_resample.shape), getsizeof(data_resample)//1024**2))
  
          rem_seq = get_sequences(np.where(hypno == 5)[0])
          # Another representation: matrix of 2 columns and n rows (n number of rem epochs), first row is the start and second is for end idx.
  
          # minimum duration > 2s
          logger.debug("STARTED: Extract REM epochs.")
          rem_seq = [(start, end) for start, end in rem_seq if (end-start) > min_dur]
          logger.debug("REM indices: {0}.".format(rem_seq))
  
          # get REM segments
          rem_idx = [(start * targetFs, (end+1) * targetFs) for start, end in rem_seq]
          rem_epochs = get_segments(rem_idx, data_resample)
          logger.debug("FINISHED: Extract REM epochs.")
  
          logger.debug("Memory usage: {0} ({1}%)".format(bytes2human(psutil.virtual_memory().available), str(psutil.virtual_memory().percent)))
  
          # Combine the REM indices with the corresponding downsampled segments
          rem = {seq:seg for seq, seg in zip(rem_seq, rem_epochs)}
  
          fname = OUTPUT_DIR + title
          logger.debug("Saving as: {0}.".format(title))
          np.save(fname, rem)
  
          del lfpHPC, hypno, data_resample, art_std, art_up, rem_seq, rem_epochs, rem
          gc.collect()

cbd_patterns = {
    "posttrial":r"[\w-]+posttrial[\w-]+",
    "hpc":"*HPC*continuous*",
    "states":"*-states*"
}

rgs_patterns = {
    "posttrial":r"[\w-]+post[\w-]+trial[\w-]+",
    "hpc":"*HPC*continuous*",
    "states":"*-states*"
}

os_patterns = {
    "posttrial":r".*post_trial.*",
    "hpc":"*HPC*",
    "states":"*states*"
}

def load_dataset(DATASET_DIR, pattern_args):
    mapped = {}

    posttrial_pattern = pattern_args["posttrial"]
    hpc_pattern = pattern_args["hpc"]
    states_pattern = pattern_args["states"]

    for root, dirs, _ in os.walk(DATASET_DIR):
        for dir in dirs:
            # Check if the directory is a post trial directory
            if re.match(posttrial_pattern, dir, flags=re.IGNORECASE):
                dir = Path(os.path.join(root, dir))
                HPC_file = next(dir.glob(hpc_pattern))
                states = next(dir.glob(states_pattern))
                mapped[str(states)] = str(HPC_file)
    return mapped

if __name__ == "__main__":
    logger = logger_setup()
    logger.debug("Saving to: {0}".format(OUTPUT_DIR))

    mapped1 = load_dataset(CBD_DIR, pattern_args=cbd_patterns)
    mapped2 = load_dataset(RGS_DIR, pattern_args=rgs_patterns)
    mapped3 = load_dataset(OS_DIR, pattern_args=os_patterns)

    print("Number of CBD recordings: ", len(mapped1))
    print("Number of RGS14 recordings: ", len(mapped2))
    print("Number of OS_Basic recordings: ", len(mapped3))

    # Wrapper for create name function for CBD dataset
    def wrapper(hpc):
        overview_df = pd.read_csv(CBD_OVERVIEW_PATH)
        return create_name_cbd(hpc, overview_df=overview_df)
    
    # CBD preprocessing
    run(mapped=mapped1, name_func=wrapper, n_down=n_down_cbd)

    # RGS14 preprocessing
    run(mapped=mapped2, name_func=create_name_rgs, n_down=n_down_rgs)

    # OS basic preprocessing
    run(mapped=mapped3, name_func=create_name_os, n_down=n_down_os)

    