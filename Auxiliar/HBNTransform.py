"""
This function builds
an Raw object from the csv files
used in the Healthy Brain Network EEG Data
http://fcon_1000.projects.nitrc.org/indi/
cmi_healthy_brain_network/downloads/downloads_EEG_R1_1.html 
"""
import mne
import numpy as np
import pandas as pd

def hbn_raw(subject,path_absolute):
    s_freq = 500

    #----------channels---------------#
    #fix later!
    #path_absolute="/home/gari/Desktop/master_tesis/notebooks/healthy_data/"

    path_subject_channels=path_absolute+subject+"_"+"RestingState_chanlocs.csv"
    print(path_subject_channels)
    #get dataframe of channles location and labels
    df_subject_channels=pd.read_csv(path_subject_channels, delimiter=",",decimal=".")

    #channels labels
    ch_labels=list(df_subject_channels.labels)

    #apply montage
    HydroCel_129 = mne.channels.make_standard_montage('GSN-HydroCel-129')

    #create info for object
    info = mne.create_info(ch_names=ch_labels, sfreq=s_freq, ch_types='eeg', montage=HydroCel_129)

    #----------signal-----------------#
    path_subject_signal=path_absolute+subject+"_"+"RestingState_data.csv"

    #put signal in microvolts
    dat_test=np.loadtxt(path_subject_signal, delimiter=',')*1e-6

    #Create the MNE Raw data object
    raw = mne.io.RawArray(dat_test, info)

    #create in stimuation channel
    stim_info = mne.create_info(['stim'], s_freq, 'stim')
    #create zero signal to store stimulus
    stim_raw = mne.io.RawArray(np.zeros(shape=[1, len(raw._times)]), stim_info)

    #add stim channle to raw signal
    raw.add_channels([stim_raw], force_update_info=True)

    #----------events-----------------#
    path_subject_events=path_absolute+subject+"_"+"RestingState_event.csv"

    #read csv of events
    df_subject_event=pd.read_csv(path_subject_events, delimiter=",",decimal=".")

    #fake structure of events
    evs = np.empty(shape=[0, 3])

    for each_element in df_subject_event.values[1:len(df_subject_event)-1]:
        if('break cnt'!=each_element[0]):
            if(int(each_element[0])==30):
                evs = np.vstack((evs, np.array([each_element[1], 0, int(each_element[0])])))
    #print(evs)
    # Add events to data object
    raw.add_events(evs, stim_channel='stim')

    #Check events
    print(mne.find_events(raw))

    #detect flat channels
    flat_chans = np.mean(raw._data[:111, :], axis=1) == 0

    # Interpolate bad channels
    # read about it here
    # https://mne.tools/dev/auto_tutorials/preprocessing/plot_15_handling_bad_channels.html

    raw.info['bads'] = list(np.array(raw.ch_names[:111])[flat_chans])
    print('Bad channels: ', raw.info['bads'])
    raw.interpolate_bads()

    # Get good eeg channel indices
    eeg_chans = mne.pick_types(raw.info, meg=False, eeg=True)

    #resample to have to 250 hz, 
    #this will allow us to compare with
    #the HDHD dataset.
    raw.resample(250, npad='auto')

    #return Raw object from mne class
    return raw,eeg_chans