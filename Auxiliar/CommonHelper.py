from scipy import signal

# Absolute path of .current script
#script_pos = os.path.dirname(os.path.abspath(__file__))

def get_index_band(rate,lower,upper):
    lower_index=int(lower*rate)
    upper_index=int(lower*rate)
    return[lower_index,upper_index]


def get_power_spectrum(X,y,channel,fs=250):
    #X=data.X
    #X=data.X[0][0,:]
    #data.X.shape=>(751, 19, 5000)
    #data.X[0][0,:].shape
    total_sample_number=X.shape[0]
    points_per_signal=X.shape[2]
    sample_holder=np.empty((1,1), dtype=float)
    for sample_number in range(0,total_sample_number):
        data_channel_holder=np.empty((1), dtype=float)
        for each_channel in range(0,channel):
            each_signal=X[sample_number,each_channel,:]
            f, Pxx_den = signal.periodogram(each_signal, fs,scaling="spectrum")
            rate_equi=(points_per_signal/fs)
            #delta power 0-4Hz
            indexs=get_index_band(rate_equi,0,4)
            delta_power=Pxx_den[indexs[0]:indexs[1]]
            #theta power 4-7hz
            indexs=get_index_band(rate_equi,4,8)
            theta_power=Pxx_den[indexs[0]:indexs[1]]
            #Alpha power 8-15hz
            indexs=get_index_band(rate_equi,8,16)
            alpha_power=Pxx_den[indexs[0]:indexs[1]]
            #beta power 16-31hz
            indexs=get_index_band(rate_equi,16,32)
            beta_power=Pxx_den[indexs[0]:indexs[1]]
            #gamma power 16-31hz
            #indexs=get_index_band(rate_equi,32,32)
            #gamma_power=Pxx_den[indexs[0]:indexs[1]]
            data_channel_holder=np.hstack([data_channel_holder,delta_power,theta_power,alpha_power,beta_power])
        sample_holder=np.vstack([sample_holder,data_channel_holder])
    return sample_holder