"""
This function builds
an Raw object from the egf files
used HDHD internal study
"""
import mne
import numpy as np
import pandas as pd


def adhd_raw(path_subject,delta_between_events_s):
    #calcute when the close eyes starts
    s_freq = 250

    #get minutes from name file
    #example: "D0000134-7-M-EO3m31s.EDF_anon.edf"
    subject=path_subject.split("/")[-1]
    min_event=int(subject.split("m")[0][-1])
    #get seconds from name file
    se_event=int(subject.split("m")[1][0:2])

    #transform to point in raw signal
    loc_time_s=min_event*60+se_event
    event_point=int(loc_time_s*s_freq)
    print("event_point",event_point)
    #fix later!
    #path_absolute="/home/gari/Desktop/master_tesis/notebooks/"  

    #----------signal-edf-----------------#
    #path_subject_signal=subject

    raw_edf=mne.io.read_raw_edf(input_fname=path_subject,
                            eog=None, 
                            misc=None, 
                            stim_channel='auto', 
                            exclude=(), 
                            preload=True, 
                            verbose=None)

    raw_edf = raw_edf.drop_channels(
                                ["AA", "LABEL"]
                            )

    #channels labels
    ch_labels=list(raw_edf.ch_names)
    ch_labels=[each_c.split(" ")[1] for each_c in ch_labels]
    print(ch_labels)

    #apply montage
    standard_10_20 = mne.channels.make_standard_montage('standard_1020')

    #create info for object
    info = mne.create_info(ch_names=ch_labels, sfreq=s_freq, ch_types='eeg', montage=standard_10_20)

    #Create the MNE Raw data object
    dat_test=raw_edf.get_data()


    #----------signal-----------------#
    raw = mne.io.RawArray(dat_test, info)

    #create in stimuation channel
    stim_info = mne.create_info(['stim'], s_freq, 'stim')
    #create zero signal to store stimulus
    stim_raw = mne.io.RawArray(np.zeros(shape=[1, len(raw._times)]), stim_info)

    #add stim channle to raw signal
    raw.add_channels([stim_raw], force_update_info=True)

    #----------events-----------------#
    evs = np.empty(shape=[0, 3])

    end_signal_point=len(raw)

    total_events=0
    add_event=True
    while add_event:
        # create delta_between_events_s interval between events
        # stop 30 seconds before ending the signal
        if (event_point+total_events*delta_between_events_s*s_freq<end_signal_point-(30*s_freq)):
            total_events=total_events+1
        else:
            add_event=False
    print("total events",total_events)

    #add events
    for i in range(0,total_events):
        step=i*delta_between_events_s*s_freq
        if i==0:
            pass
            #evs = np.vstack((evs, np.array([event_point, 0, int(90)])))
            #print(evs)
        else:
            acc=event_point+step
            evs=np.vstack((evs, np.array([acc, 0, int(60)])))


    raw.add_events(evs, stim_channel='stim')

    #detect flat channels
    flat_chans = np.mean(raw._data[:19, :], axis=1) == 0

    # Interpolate bad channels
    # read about it here
    # https://mne.tools/dev/auto_tutorials/preprocessing/plot_15_handling_bad_channels.html

    raw.info['bads'] = list(np.array(raw.ch_names[:19])[flat_chans])
    print('Bad channels: ', raw.info['bads'])
    raw.interpolate_bads()

    # Get good eeg channel indices
    eeg_chans = mne.pick_types(raw.info, meg=False, eeg=True)



    
    return raw,eeg_chans