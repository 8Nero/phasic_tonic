import re
from pathlib import Path
import logging

import numpy as np
from scipy.signal import hilbert
from neurodsp.filt import filter_signal

logger = logging.getLogger('runtime')

def get_sequences(x, ibreak=1):
    """
    Identifies contiguous sequences.

    Parameters:
    x (np.ndarray): 1D time series.
    ibreak (int): A threshold value for determining breaks between sequences (default is 1).

    Returns:
    list of tuples: Each tuple contains the start and end integer of each contiguous sequence.
    """
    if len(x) == 0:
        return []

    diff = np.diff(x)
    breaks = np.where(diff > ibreak)[0]

    # Append the last index to handle the end of the array
    breaks = np.append(breaks, len(x) - 1)
    
    sequences = []
    start_idx = 0
    
    for break_idx in breaks:
        end_idx = break_idx
        sequences.append((x[start_idx], x[end_idx]))
        start_idx = end_idx + 1
    
    return sequences

def get_segments(idx, signal):
    """
    Extracts segments of the signal between specified start and end time indices.

    Parameters:
    idx (list of tuples): Each tuple contains (start_time, end_time).
    signal (np.ndarray): The signal from which to extract segments.

    Returns:
    list of np.ndarray: Each element is a segment of the signal corresponding to the given time ranges.
    """
    segments = []
    for (start_time, end_time) in idx:
        if end_time > len(signal):
            end_time = len(signal) - 1
        segment = signal[start_time:end_time]
        segments.append(segment)
    
    return segments

def _detect_troughs(signal, thr):
    lidx  = np.where(signal[0:-2] > signal[1:-1])[0]
    ridx  = np.where(signal[1:-1] <= signal[2:])[0]
    thidx = np.where(signal[1:-1] < thr)[0]
    sidx = np.intersect1d(lidx, np.intersect1d(ridx, thidx))+1
    return sidx

def phasic_detect(rem, fs, thr_dur=900, nfilt=11):
    w1 = 5.0
    w2 = 12.0
    nfilt = 11
    thr_dur = 900

    trdiff_list = []
    rem_eeg = np.array([])
    eeg_seq = {}
    sdiff_seq = {}
    tridx_seq = {}
    filt = np.ones((nfilt,))
    filt = filt / filt.sum()

    for idx in rem:
        start, end = idx

        epoch = rem[idx]
        epoch = filter_signal(epoch, fs, 'bandpass', (w1,w2), remove_edges=False)
        epoch = hilbert(epoch)

        inst_phase = np.angle(epoch)
        inst_amp = np.abs(epoch)

        # trough indices
        tridx = _detect_troughs(inst_phase, -3)

        # alternative version:
        #tridx = np.where(np.diff(np.sign(np.diff(eegh))))[0]+1

        # differences between troughs
        trdiff = np.diff(tridx)

        # smoothed trough differences
        sdiff_seq[idx] = np.convolve(trdiff, filt, 'same')

        # dict of trough differences for each REM period
        tridx_seq[idx] = tridx

        eeg_seq[idx] = inst_amp

        # differences between troughs
        trdiff_list += list(trdiff)

        # amplitude of the entire REM sleep
        rem_eeg = np.concatenate((rem_eeg, inst_amp)) 

    trdiff = np.array(trdiff_list)
    trdiff_sm = np.convolve(trdiff, filt, 'same')

    # potential candidates for phasic REM:
    # the smoothed difference between troughs is less than
    # the 10th percentile:
    thr1 = np.percentile(trdiff_sm, 10)
    # the minimum difference in the candidate phREM is less than
    # the 5th percentile
    thr2 = np.percentile(trdiff_sm, 5)
    # the peak amplitude is larger than the mean of the amplitude
    # of the REM EEG.
    thr3 = rem_eeg.mean()

    logger.debug("Thresholds: thr1 = {0:.3f}, thr2 = {1:.3f}, thr3 = {2:.3f}".format(thr1, thr2, thr3))

    phrem = {}
    for rem_idx in tridx_seq:
        rem_start, rem_end = rem_idx
        offset = rem_start * fs

        phrem[rem_idx] = []

        # trough indices
        tridx = tridx_seq[idx]

        # smoothed trough interval
        sdiff = sdiff_seq[idx]

        # ampplitude of the REM epoch
        eegh = eeg_seq[idx]

        cand_idx = np.where(sdiff <= thr1)[0]
        cand = get_sequences(cand_idx)

        logger.debug("Candidates: {0}".format(str(cand)))
        for start, end in cand:
            dur = ( (tridx[end]-tridx[start]+1)/fs ) * 1000
            if dur > thr_dur and np.min(sdiff[start:end]) < thr2 and np.mean(eegh[tridx[start]:tridx[end]+1]) > thr3:
                a = tridx[start]   + offset
                b = tridx[end]  + offset
                
                if b > (rem_end * fs):
                    b = rem_end*fs

                ph_idx = (a,b)
                phrem[rem_idx].append(ph_idx)
    return phrem

def create_name_cbd(file, overview_df):
    #pattern for matching the information on the rat
    pattern = r'Rat(\d+)_.*_SD(\d+)_([A-Z]+).*posttrial(\d+)'

    # extract the information from the file path
    match = re.search(pattern, file)
    rat_num = int(match.group(1))
    sd_num = int(match.group(2))
    condition = str(match.group(3))
    posttrial_num = int(match.group(4))

    mask = (overview_df['Rat no.'] == rat_num) & (overview_df['Study Day'] == sd_num) & (overview_df['Condition'] == condition)

    # use boolean indexing to extract the Treatment value
    treatment_value = overview_df.loc[mask, 'Treatment'].values[0]
    
    # Extract the value from the "treatment" column of the matching row
    if treatment_value == 0:
        treatment = '0'
    else:
        treatment = '1'
       
    title_name = 'Rat' + str(rat_num) +'_' + 'SD' + str(sd_num) + '_' + condition + '_' + treatment + '_' + 'posttrial' + str(posttrial_num)
    #RatID,StudyDay,condition,conditionfull, treatment, treatmentfull, posstrial number

    return title_name

def create_name_rgs(fname):
    #pattern for matching the information on the rat
    pattern = r'Rat(\d+)_.*_SD(\d+)_([A-Z]+).*post[\w-]+trial(\d+)'
    
    # extract the information from the file path
    match = re.search(pattern, fname, flags=re.IGNORECASE)
    rat_num = int(match.group(1))
    sd_num = int(match.group(2))
    condition = str(match.group(3))
    posttrial_num = int(match.group(4))

    # Extract the value from the "treatment" column of the matching row
    if (rat_num == 1) or (rat_num == 2) or (rat_num == 6) or (rat_num == 9) :
        treatment = '2'
    else:
        treatment = '3'
    
    title_name = 'Rat' + str(rat_num) +'_' + 'SD' + str(sd_num) + '_' + condition + '_' + treatment + '_' + 'posttrial' + str(posttrial_num)
    #RatID,StudyDay,condition,conditionfull, treatment, treatmentfull, posstrial number

    return title_name


def create_name_os(hpc_fname):
    metadata = str(Path(hpc_fname).parent.parent.name).split("_")
    title = metadata[1] + "_" + metadata[2] + "_" + metadata[3]
    
    pattern = r"post_trial(\d+)"
    match = re.search(pattern, hpc_fname, re.IGNORECASE)
    title += "_4_" + "posttrial" + match.group(1)

    #RatID,StudyDay,condition,conditionfull, treatment, treatmentfull, posstrial number
    return title