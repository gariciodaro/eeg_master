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



