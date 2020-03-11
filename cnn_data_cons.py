import logging

import pandas as pd
import scipy
import scipy.signal

import sys
import os
import mne
import numpy as np

#/home/gari/anaconda3/envs/EEG/lib/python3.7/site-packages/Braindecode-0.4.85-py3.7.egg/braindecode/datautil

# Absolute path of .current script
script_pos = os.path.dirname(os.path.abspath(__file__))
#script_pos = os.path.dirname(__file__)

#print("script_pos",script_pos)

Auxiliar_pos=script_pos+"/Auxiliar"

# Include Auxiliar_pos in the current python enviroment
if not Auxiliar_pos in sys.path:
    sys.path.append(Auxiliar_pos)


Data_pos=script_pos+"/Data"
# Include Data_pos in the current python enviroment
#if not Data_pos in sys.path:
#    sys.path.append(Data_pos)
Picke_pos=script_pos+"/picke_files"

# My custom package for transforming the 
# healthy database csv files to
# mne objects
import HBNTransform as HBT

# My custom package for transforming the 
# edf signas of HDHD patietns to
# mne objects
import ADHDTransform as ADHD

from OFHandlers import OFHandlers as OFH

import PairSignalConcat




def exponential_running_standardize(
    data, factor_new=0.001, init_block_size=None, eps=1e-4
):
    """
    Perform exponential running standardization. 
    
    Compute the exponental running mean :math:`m_t` at time `t` as 
    :math:`m_t=\mathrm{factornew} \cdot mean(x_t) + (1 - \mathrm{factornew}) \cdot m_{t-1}`.
    
    Then, compute exponential running variance :math:`v_t` at time `t` as 
    :math:`v_t=\mathrm{factornew} \cdot (m_t - x_t)^2 + (1 - \mathrm{factornew}) \cdot v_{t-1}`.
    
    Finally, standardize the data point :math:`x_t` at time `t` as:
    :math:`x'_t=(x_t - m_t) / max(\sqrt{v_t}, eps)`.
    
    
    Parameters
    ----------
    data: 2darray (time, channels)
    factor_new: float
    init_block_size: int
        Standardize data before to this index with regular standardization. 
    eps: float
        Stabilizer for division by zero variance.

    Returns
    -------
    standardized: 2darray (time, channels)
        Standardized data.
    """
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    squared = demeaned * demeaned
    square_ewmed = squared.ewm(alpha=factor_new).mean()
    standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
    standardized = np.array(standardized)
    if init_block_size is not None:
        other_axis = tuple(range(1, len(data.shape)))
        init_mean = np.mean(
            data[0:init_block_size], axis=other_axis, keepdims=True
        )
        init_std = np.std(
            data[0:init_block_size], axis=other_axis, keepdims=True
        )
        init_block_standardized = (
            data[0:init_block_size] - init_mean
        ) / np.maximum(eps, init_std)
        standardized[0:init_block_size] = init_block_standardized
    return standardized

def bandpass_cnt(
    data, low_cut_hz, high_cut_hz, fs, filt_order=3, axis=0, filtfilt=False
):
    """
     Bandpass signal applying **causal** butterworth filter of given order.

    Parameters
    ----------
    data: 2d-array
        Time x channels
    low_cut_hz: float
    high_cut_hz: float
    fs: float
    filt_order: int
    filtfilt: bool
        Whether to use filtfilt instead of lfilter

    Returns
    -------
    bandpassed_data: 2d-array
        Data after applying bandpass filter.
    """
    if (low_cut_hz == 0 or low_cut_hz is None) and (
        high_cut_hz == None or high_cut_hz == fs / 2.0
    ):
        log.info(
            "Not doing any bandpass, since low 0 or None and "
            "high None or nyquist frequency"
        )
        return data.copy()
    if low_cut_hz == 0 or low_cut_hz == None:
        log.info("Using lowpass filter since low cut hz is 0 or None")
        return lowpass_cnt(
            data, high_cut_hz, fs, filt_order=filt_order, axis=axis
        )
    if high_cut_hz == None or high_cut_hz == (fs / 2.0):
        log.info(
            "Using highpass filter since high cut hz is None or nyquist freq"
        )
        return highpass_cnt(
            data, low_cut_hz, fs, filt_order=filt_order, axis=axis
        )

    nyq_freq = 0.5 * fs
    low = low_cut_hz / nyq_freq
    high = high_cut_hz / nyq_freq
    b, a = scipy.signal.butter(filt_order, [low, high], btype="bandpass")
    assert filter_is_stable(a), "Filter should be stable..."
    if filtfilt:
        data_bandpassed = scipy.signal.filtfilt(b, a, data, axis=axis)
    else:
        data_bandpassed = scipy.signal.lfilter(b, a, data, axis=axis)
    return data_bandpassed


def filter_is_stable(a):
    """
    Check if filter coefficients of IIR filter are stable.
    
    Parameters
    ----------
    a: list or 1darray of number
        Denominator filter coefficients a.

    Returns
    -------
    is_stable: bool
        Filter is stable or not.  
    Notes
    ----
    Filter is stable if absolute value of all  roots is smaller than 1,
    see [1]_.
    
    References
    ----------
    .. [1] HYRY, "SciPy 'lfilter' returns only NaNs" StackOverflow,
       http://stackoverflow.com/a/8812737/1469195
    """
    assert a[0] == 1.0, (
        "a[0] should normally be zero, did you accidentally supply b?\n"
        "a: {:s}".format(str(a))
    )
    # from http://stackoverflow.com/a/8812737/1469195
    return np.all(np.abs(np.roots(a)) < 1)

path_ADHD_data=Data_pos+"/ADHD_data/children_edf_anon/"
ADHD_subjects=os.listdir(path_ADHD_data)

print("total ADHD_subjects available",len(ADHD_subjects))

selected_ADHD_subjects=OFH.load_object("./selected_ADHD_subjects.file")

"""
for each_adhd in selected_ADHD_subjects[0:73]:
    

    print(path_ADHD_data+each_adhd)
    raw_ADHD,eeg_chans=ADHD.adhd_raw(path_subject=path_ADHD_data+each_adhd,delta_between_events_s=20)
    dat_evs_ADHD = mne.find_events(raw_ADHD)
    raw_ADHD.set_eeg_reference(ref_channels='average',projection=True)
    ec_epochs = mne.Epochs(raw_ADHD, events=dat_evs_ADHD, event_id={'EC': 60}, tmin=0.0, tmax= 20,
                        baseline=None, picks=eeg_chans, preload=True)
    #ec_epochs.plot_psd_topomap()
    #user_input=input("pick signas (y/n) :")
    #if user_input=="y":
"""

map_monstages=OFH.load_object(Picke_pos+"/map_monstages.file")
keys_129=OFH.load_object(Picke_pos+"/keys_129.file")
keys_1020=OFH.load_object(Picke_pos+"/keys_1020.file")

path_healthy_data=Data_pos+"/healthy_data/"
#NDARAA075AMK_RestingState_chanlocs
healthy_subjects=[ file.split("_")[0] for file in os.listdir(path_healthy_data) if "chanlocs" in file]
#print(healthy_subjects[0:19])
print("total healthy_subjects available",len(healthy_subjects))
#DARAC904DMU
i=0
for each_healthy in healthy_subjects[0:65]:
    
    print("*"*100)
    print(i)
    raw_healthy,eeg_chans_he=HBT.hbn_raw(subject=each_healthy,path_absolute=path_healthy_data)

    dat_evs_he = mne.find_events(raw_healthy)

    mne.rename_channels(raw_healthy.info,map_monstages)

    list_to_drop=[each_key for each_key in raw_healthy.ch_names if each_key not in keys_1020+['stim']]
    raw_healthy.drop_channels(list_to_drop)

    # ADHD
    each_adhd=selected_ADHD_subjects[i]
    print(path_ADHD_data+each_adhd)
    raw_ADHD,eeg_chans=ADHD.adhd_raw(path_subject=path_ADHD_data+each_adhd,delta_between_events_s=20)
    #dat_evs_ADHD = mne.find_events(raw_ADHD)
    #raw_ADHD.set_eeg_reference(ref_channels='average',projection=True)
    #ec_epochs = mne.Epochs(raw_ADHD, events=dat_evs_ADHD, event_id={'EC': 60}, tmin=0.0, tmax= 20,
    #                   baseline=None, picks=eeg_chans, preload=True)

    #merge signals

    #concat_raw=mne.concatenate_raws([raw_healthy,raw_ADHD])
    signal_healthy=PairSignalConcat.concat_prepare_cnn(raw_healthy)
    signal_ADHD=PairSignalConcat.concat_prepare_cnn(raw_ADHD)
    if(i==0):
        print(i,each_healthy,each_adhd)
        concat_signal=signal_healthy
        concat_signal.X=np.vstack([signal_healthy.X,signal_ADHD.X])
        concat_signal.y=np.concatenate((signal_healthy.y,signal_ADHD.y), axis=0)
        #concat_signal=PairSignalConcat.concat_prepare_cnn(raw_healthy)
        print("concat_signal.X.shape",concat_signal.X.shape)
        print("concat_signal.y.shape",concat_signal.y.shape)
    else:
        print("+"*100)
        print(i,each_healthy,each_adhd)
        concat_signal.X=np.vstack([concat_signal.X,signal_healthy.X,signal_ADHD.X])
        concat_signal.y=np.concatenate((concat_signal.y,signal_healthy.y,signal_ADHD.y), axis=0)
        #concat_signal=PairSignalConcat.concat_prepare_cnn(raw_healthy)
        print("concat_signal.X.shape",concat_signal.X.shape)
        print("concat_signal.y.shape",concat_signal.y.shape)
        """
        print("concat_signal.X.shape",concat_signal.X.shape)
        print("concat_signal.y.shape",concat_signal.y.shape)
        incoming_data=PairSignalConcat.concat_prepare_cnn(concat_raw)
        print("incoming_data.X.shape",incoming_data.X.shape)
        print("incoming_data.y.shape",incoming_data.y.shape)
        concat_signal.X=np.vstack([concat_signal.X,incoming_data.X])
        concat_signal.y=np.concatenate((concat_signal.y,incoming_data.y), axis=0)
        #concat_signal.y=np.hstack([concat_signal.y,incoming_data.y])
        """

    i=i+1
print("concat_signal.X.shape",concat_signal.X.shape)

"""
ev,ch,points=concat_signal.X.shape
#(8, 19, 5000)
temp_signal=concat_signal.X.reshape(ch,ev*points)

temp_signal=bandpass_cnt(temp_signal,
                            low_cut_hz,
                            high_cut_hz,
                            250,
                            filt_order=3,
                            axis=1)

temp_signal=exponential_running_standardize(temp_signal,
            factor_new=factor_new,
            init_block_size=init_block_size,
            eps=1e-4,
        )
temp_signal=temp_signal.reshape(ev,ch,points)
concat_signal.X=temp_signal
"""
OFH.save_object("./concat_signal.file",concat_signal)



