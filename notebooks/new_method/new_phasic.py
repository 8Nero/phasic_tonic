#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:21:51 2024

@author: abdelrahmanrayan
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from mne.filter import resample
import yasa
import seaborn as sns

# neurodigital signal processing toolbox 
from neurodsp.filt import filter_signal
from neurodsp.plts import plot_time_series
from neurodsp.sim import sim_combined
# Import utilities for loading and plotting data
from neurodsp.utils import create_times
from neurodsp.utils.download import load_ndsp_data
from neurodsp.plts.time_series import plot_time_series, plot_instantaneous_measure
# Import time-frequency functions
from neurodsp.timefrequency import amp_by_time, freq_by_time, phase_by_time
from neurodsp.plts import plot_time_series, plot_bursts

# scipy library 
import scipy.io
from scipy.signal import hilbert

# cycle by cycle library 
from bycycle.features import compute_features
from bycycle.cyclepoints import find_extrema, find_zerox
from bycycle.cyclepoints.zerox import find_flank_zerox
from bycycle.plts import plot_burst_detect_summary, plot_cyclepoints_array
from bycycle.utils.download import load_bycycle_data
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

def _despine_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

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
def phasic_rem_v3(eeg, hypno, sr,min_dur=2.5, vm = 3000, thr_dur = 900, pplot=False, nfilt=11):
    """
    Detect phasic REM episodes using the algorithm described in 
    Daniel Gomes de Almeida‐Filho et al. 2021, which comes from
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
    from scipy.signal import hilbert

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
#        tb = s[-1]*(nbin+1)
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
                idx = range(a,b+1)
    
                if si in phrem:
                    phrem[si].append(idx)
                else:
                    phrem[si] = [idx]
    
    # make plot:
    if pplot:
        nsr_seg = 1 # before 1
        # overlap of consecutive FFT windows
        perc_overlap = 0.8
        

        freq, t, SP = scipy.signal.spectrogram(EEG, fs=sr, window='hann', nperseg=int(nsr_seg * sr),
                                                   noverlap=int(nsr_seg * sr * perc_overlap))
            # for nsr_seg=1 and perc_overlap = 0.9,
            # t = [0.5, 0.6, 0.7 ...]
        dt = t[1]-t[0]
            # Note: sp_name.mat includes keys: SP, SP2, freq, dt, t
    

        SP   = SP
        freq = freq
        tbs    = t
        dt   = dt
        
        plt.figure()
        # plot spectrogram
        ax = plt.subplot(512)
        ifreq = np.where(freq <= 20)[0]
        ax.pcolorfast(tbs, freq[ifreq], SP[ifreq,:], vmin=0, vmax=vm, cmap='jet')
        plt.ylabel('Freq. (Hz)')
        
        # plot hypnogram
        axes_brs = plt.subplot(511, sharex=ax)
        cmap = plt.cm.jet
        my_map = cmap.from_list('brs', [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8]], 5)
    
        tmp = axes_brs.pcolorfast(tbs, [0, 1], np.array([M]), vmin=1, vmax=5)
        tmp.set_cmap(my_map)
        axes_brs.axis('tight')
        _despine_axes(axes_brs)
        plt.xlim([tbs[0], tbs[-1]])

    
        # plot gamma power
        plt.subplot(513, sharex=ax)
        gamma = [50, 90]
        df = freq[1] - freq[0]
        igamma = np.where((freq >= gamma[0]) & (freq <= gamma[1]))[0]
        pow_gamma = SP[igamma,:].sum(axis=0) * df
        plt.plot(tbs, pow_gamma)
        plt.xlim([tbs[0], tbs[-1]])
        plt.ylabel(r'$\gamma$')

        # plot theta/delta
        #plt.subplot(514, sharex=ax)
        # theta = [6, 12]
        # delta = [0.5, 4.5]
        # itheta = np.where((freq >= theta[0]) & (freq <= theta[1]))[0]
        # idelta = np.where((freq >= delta[0]) & (freq <= delta[1]))[0]
        # pow_theta = SP[itheta,:].sum(axis=0) * df
        # pow_delta = SP[idelta,:].sum(axis=0) * df        
        # plt.plot(tbs, np.divide(pow_theta, pow_delta))
        # plt.xlim([tbs[0], tbs[-1]])
        
        # plot raw EEG; downsample for faster plotting
        plt.subplot(515, sharex=axes_brs)
        EEGdn = downsample_vec(EEG, 4)
        teeg = np.arange(0, len(EEG)) * (1/sr)
        teeg_dn = np.arange(0, len(EEGdn)) * ((1/sr)*4)

        for tr in tridx_list:            
            idx = range(tr[0], tr[-1]+1)
            idx_dn = [int(i/4) for i in idx]
            
            eeg = EEGdn[idx_dn]                        
            plt.plot(teeg_dn[idx_dn], eeg, 'k')        
        plt.xlim([0, teeg[-1]])
        
        for si in phrem:
            ta =  si*nbin
            
            idx_list = phrem[si]
            eegh = eeg_seq[si]
            sdiff = sdiff_seq[si]

            # plot amplitude
            plt.plot(teeg[ta:ta+len(eegh)], eegh, 'g')
            # plot threshold for amplitude
            plt.plot([teeg[ta], teeg[ta+len(eegh)-1]], [thr3, thr3], 'r--')

            for idx in idx_list:
                a = idx[0]
                b = idx[-1]
                a = int(a/4)
                b = int(b/4)
                
                plt.plot(teeg_dn[range(a,b+1)], EEGdn[a:b+1], 'r')
        plt.ylabel('EEG')
        # plot laser
#        lsr = load_laser(ppath, name)
#        plt.plot(teeg, lsr*500, 'blue')
                
        # plot smoothed inter-through intervals
        plt.subplot(514, sharex=ax)
        for si in phrem:
            ta = si*nbin

            tridx = tridx_seq[si] + ta                        
            sdiff = sdiff_seq[si]
            plt.plot(teeg[tridx[:-1]], sdiff, 'k')
            plt.plot(teeg[[tridx[0], tridx[-1]]], [thr2, thr2], 'r')
            plt.plot(teeg[[tridx[0], tridx[-1]]], [thr1, thr1], 'b')
        plt.ylabel('ITIs')
        
    return phrem
# tonic indices, duration of tonic and phasic, count of tonic and phasic, count normalized by total duration (frequency) for both tonic and phasic 
# post trial 1, 2, 3 and 4 are 45 minutes
# post trial 5 is 3 hours which should be 180 minutes 
#%% directory for test data 
# upload one dataset from the pfc 
os.chdir('/Volumes/MacBack/GenzelLab/theta_gamma_the_roads_paper/OS_basic_separated/1/Rat-OS-Ephys_Rat1_SD4_CON_28-09-2017/post_trial5_2017-09-28_14-55-18')


#%%

# reading the pfc data 
lfpHPC = scipy.io.loadmat('HPC_100_CH46.continuous.mat')['HPC']
lfpHPC = lfpHPC.flatten()

# read the states 
hypno = scipy.io.loadmat('post_trial5_2017-09-28_14-55-18-statesAlysha.mat')['states']
hypno = hypno.flatten()
#%%
fs = 2500
#time_vect = create_times(n_sec-1, fs)
time_vect = np.arange(0, len(lfpHPC)/fs, 1/fs)
#%% downsample the data 
targetFs = 500
n_down = fs/targetFs
data_resample = resample(lfpHPC, down=n_down, method='fft')
time_vect2 = np.arange(0, len(data_resample)/targetFs, 1/targetFs)
#%%
plt.plot(time_vect2, data_resample)
#%%
art_std, zscores_std = yasa.art_detect(data_resample,targetFs , window=1, method='std', threshold=4, verbose='info')
art_up = yasa.hypno_upsample_to_data(art_std, 1, data_resample, targetFs)
#%% checking the noise outcome 

# Plot the artifact vector
plt.plot(art_up);
plt.yticks([0, 1], labels=['Good (0)', 'Art (1)']);

#%%
sns.distplot(zscores_std)
plt.title('Histogram of z-scores')
plt.xlabel('Z-scores')
plt.ylabel('Density')
plt.axvline(3, color='r', label='Threshold')
plt.axvline(-3, color='r')
plt.legend(frameon=False);
#%%
plt.plot(zscores_std)
plt.axhline(y=4)
#%% plotting the artefacts 
data_resample[art_up] = 0
#%%
plot_bursts(time_vect2, data_resample, art_up, lw=2,
                labels=['Raw Data', 'Artefacts'], xlabel='Time [s]', ylabel='Amplitude [$\mu$V]', figsize= (8,4))
#plt.xlim((2460,2528))
plt.show()
#%%
phasecREMfreq= phasic_rem_v3(data_resample-np.mean(data_resample), hypno, targetFs,min_dur=2.5, vm = 1000, thr_dur = 900, pplot=True, plaser=False, nfilt=11)