import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

# neurodigital signal processing toolbox 
from neurodsp.filt import filter_signal
from neurodsp.sim import sim_combined

# Import time-frequency functions
from neurodsp.timefrequency import amp_by_time, freq_by_time, phase_by_time
from neurodsp.plts import plot_time_series, plot_bursts

# scipy library 
import scipy.io
from scipy.signal import hilbert

#%% relevant functions for the analysis 
def get_sequences(idx, ibreak=1) :  
    """
    get_sequences(idx, ibreak=1)
    idx     -    np.vector of indices
    @RETURN:
    seq     -    list of np.vectors
    """
    diff = idx[1:] - idx[0:-1]
    breaks = np.nonzero(diff>ibreak)[0]
    breaks = np.append(breaks, len(idx)-1)
    
    seq = []    
    iold = 0
    for i in breaks:
        r = list(range(iold, i+1))
        seq.append(idx[r])
        iold = i+1
        
    return seq
def my_bpfilter(x, w0, w1, N=4,bf=True):
    """
    create N-th order bandpass Butterworth filter with corner frequencies 
    w0*sampling_rate/2 and w1*sampling_rate/2
    """
    #from scipy import signal
    #taps = signal.firwin(numtaps, w0, pass_zero=False)
    #y = signal.lfilter(taps, 1.0, x)
    #return y
    from scipy import signal
    b,a = signal.butter(N, [w0, w1], 'bandpass')
    if bf:
        y = signal.filtfilt(b,a, x)
    else:
        y = signal.lfilter(b,a, x)
        
    return y

def _detect_troughs(signal, thr):
    lidx  = np.where(signal[0:-2] > signal[1:-1])[0]
    ridx  = np.where(signal[1:-1] <= signal[2:])[0]
    thidx = np.where(signal[1:-1] < thr)[0]
    sidx = np.intersect1d(lidx, np.intersect1d(ridx, thidx))+1
    return sidx

def downsample_vec(x, nbin):
    """
    y = downsample_vec(x, nbin)
    downsample the vector x by replacing nbin consecutive \
    bin by their mean \
    @RETURN: the downsampled vector 
    """
    n_down = int(np.floor(len(x) / nbin))
    x = x[0:n_down*nbin]
    x_down = np.zeros((n_down,))

    # 0 1 2 | 3 4 5 | 6 7 8 
    for i in range(nbin) :
        idx = list(range(i, int(n_down*nbin), int(nbin)))
        x_down += x[idx]

    return x_down / nbin

def phasic_rem_v3(eeg, hypno, sr, min_dur=2.5, vm = 3000, thr_dur = 900, nfilt=11):
        """
        Detect phasic REM episodes using the algorithm described in 
        Daniel Gomes de Almeida-Filho et al. 2021, which comes from
        https://www.nature.com/articles/nn.2894 Mizuseki et al. 2011
        

        Parameters
        ----------
        ppath : TYPE
            DESCRIPTION.
        name : TYPE
            DESCRIPTION.
        min_dur : TYPE, optional
            DESCRIPTION. The default is 2.5.
        plaser : bool, optional
            if True, only use REM states w/o laser to set thresholds for the algorithm

        Returns
        -------
        phrem : dict
            dict: start index of each REM episode in hypnogram --> all sequences of phasic REM episodes;
            not that phREM sequences are represented as indices in the raw EEG

        """
        M = hypno
        EEG = eeg
        neeg = EEG.shape[0]
        seq = get_sequences(np.where(M == 5)[0])
        rem_idx = []
        for s in seq:
            rem_idx += list(s)
        
        sr = sr
        nbin = sr
        sdt = nbin*(1/sr)
        
        w1 = 5.0
        w2 = 12.0
        
        filt = np.ones((nfilt,))
        filt = filt / filt.sum()
        
        trdiff_list = []
        tridx_list = []
        rem_eeg = np.array([])
        eeg_seq = {}
        sdiff_seq = {}
        tridx_seq = {}
        
        # Collect for each REM sequence the smoothed inter-trough intervals
        # and EEG amplitudes as well as the indices of the troughs.
        seq = [s for s in seq if len(s)>=min_dur]
        for s in seq:
            ta = s[0]*nbin
            #tb = s[-1]*(nbin+1)
            tb = (s[-1]+1)*nbin
            tb = np.min((tb, neeg))
                    
            eeg_idx = np.arange(ta, tb) # this the whole REM epoch       
            eeg = EEG[eeg_idx]    
            if len(eeg)*(1/sr) <= min_dur:
                continue

            eegh =  filter_signal(eeg, sr, 'bandpass',(w1,w2), remove_edges=False)
            res = hilbert(eegh)
            instantaneous_phase = np.angle(res)
            amp = np.abs(res)
        
            # trough indices
            tridx = _detect_troughs(instantaneous_phase, -3)
            # Alternative that does not seems to work that well:        
            #tridx = np.where(np.diff(np.sign(np.diff(eegh))))[0]+1
            
            # differences between troughs
            trdiff = np.diff(tridx)
        
            # smoothed trough differences
            sdiff_seq[s[0]] = np.convolve(trdiff, filt, 'same')

            # dict of trough differences for each REM period
            tridx_seq[s[0]] = tridx
            
            eeg_seq[s[0]] = amp
        
        rem_idx = []    
        for s in seq:
            rem_idx += list(s)
        


        # collect again smoothed inter-trough differences and amplitude;
        # but this time concat the data to one long vector each (@trdiff_sm and rem_eeg)
        for s in seq:
            ta = s[0]*nbin
            tb = (s[-1]+1)*nbin
            tb = np.min((tb, neeg))

            eeg_idx = np.arange(ta, tb)
            eeg = EEG[eeg_idx]            
            if len(eeg)*(1/sr) <= min_dur:
                continue
            
            eegh = filter_signal(eeg, sr, 'bandpass',(w1,w2), remove_edges=False)
            res = hilbert(eegh)
            instantaneous_phase = np.angle(res)
            amp = np.abs(res)
        
            # trough indices
            tridx = _detect_troughs(instantaneous_phase, -3)
            # alternative version:
            #tridx = np.where(np.diff(np.sign(np.diff(eegh))))[0]+1

            # differences between troughs
            tridx_list.append(tridx+ta)
            trdiff = np.diff(tridx)
            trdiff_list += list(trdiff)
        
            rem_eeg = np.concatenate((rem_eeg, amp)) 
        
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

        phrem = {}
        for si in tridx_seq:
            offset = nbin*si
            
            tridx = tridx_seq[si]
            sdiff = sdiff_seq[si]
            eegh = eeg_seq[si]
            
            idx = np.where(sdiff <= thr1)[0]
            cand = get_sequences(idx)
        
            #thr4 = np.mean(eegh)    
            for q in cand:
                dur = ( (tridx[q[-1]]-tridx[q[0]]+1)/sr ) * 1000
                #if 16250 > si*nbin * (1/sr) > 16100:
                #    print((tridx[q[0]]+si*nbin) * (1/sr))

                if dur > thr_dur and np.min(sdiff[q]) < thr2 and np.mean(eegh[tridx[q[0]]:tridx[q[-1]]+1]) > thr3:
                    
                    a = tridx[q[0]]   + offset
                    b = tridx[q[-1]]  + offset
                    idx = (a,b)
        
                    if si in phrem:
                        phrem[si].append(idx)
                    else:
                        phrem[si] = [idx]
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