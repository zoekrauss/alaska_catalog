import obspy
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import pandas as pd
import time; import datetime
from obspy.core.utcdatetime import UTCDateTime
from obspy.clients.fdsn.client import Client
import pyasdf
import scipy.fftpack

def read_files(base_folder,use_quakeml=False):
    ''' 
    Reads in earthquake catalog files from CSS format into useful pandas dataframes

    Inputs:
    base_folder = string containing name of year and month of interest, e.g. '2019_01'

    Outputs dataframes:
    assoc
    arrivals
    origin
    '''

    
    base_dir = 'catalog_css/'+base_folder+'/'
    
    arr_files = glob(base_dir + 'catalog_XO_*arrival')
    assoc_files = glob(base_dir + 'catalog_XO_*assoc')
    origin_files = glob(base_dir + 'catalog_XO_*origin')
    
    
    # Read data into pandas dataframe
    arrivals = pd.concat([pd.read_csv(f,header=None,delim_whitespace=True) for f in arr_files])
    assoc = pd.concat([pd.read_csv(f,header=None,delim_whitespace=True) for f in assoc_files])
    origin = pd.concat([pd.read_csv(f,header=None,delim_whitespace=True) for f in origin_files])

    # Rename some columns for clarity:
    assoc=assoc.rename(columns={0: "arrivalid", 1: "originid",2:"stationcode",3:"phase"})
    origin = origin.rename(columns={4:'originid',3:'epochtime',20:'magnitude'})
    arrivals=arrivals.rename(columns={2: "arrivalid", 6: "channel",0:"stationcode", 1:'epochtime',7:'phase'})
    
    # Read in quakeml to get the unique event identifiers
    if use_quakeml:
        quake_ml = base_dir+'*.quakeml'
        cat = obspy.core.event.read_events(quake_ml)
        # Match up all origins to quakeml event ids
        quakeml_id=[]
        for i in range(0,len(origin)):
            quake = origin.iloc[i]
            otime = UTCDateTime(datetime.datetime.utcfromtimestamp(quake['epochtime']))
            for event in cat:
                timediff=otime-event.origins[0].time
                if timediff<1:
                    quakeml_id.append(str(event.origins[0].resource_id)[37:])
                    break
        # Add quakeml ids to origin dataframe
        origin['quakeml_id']=quakeml_id
    
    return(assoc,arrivals,origin)

def calc_snr(data,sampleind,sr,phase):
    """
    # Calculate SNR of arrival
    # INPUTS:
    # data = seismic data in a vector; such as trace.data from an obspy stream
    # sampleind = index in the data of desired arrival for which to calculate SNR
    # sr = sampling rate of data
    # phase = type of arrival as a string, either 'P' or 'S'
    #
    # OUTPUT:
    # snr = float object of calculated SNR for the input index
    """

    
    if phase == 'P':
        window = [5,5] # in seconds
    if phase == 'S':
        window = [5,5]
    try:
        sr = int(sr)
        snr_num = max(abs(data[sampleind:(sampleind+(window[0]*sr))]))
        snr_denom = np.sqrt(np.mean((data[(sampleind-(window[1]*sr)):sampleind])**2))
        snr = snr_num/snr_denom
    except:
        snr = float('NaN')
    return snr

def get_stationlist(arrivals,assoc,origin,index=None):
    """
    Go through events in a given month and get list of stations that have both a P- and S-wave pick
    
    inputs:
    arrivals - pandas dataframe containing arrival information as produced by read_files
    assoc - pandas dataframe that links arrival and origin information as produced by read_files
    origin - pandas dataframe containing earthquake information as produced by read_files
    index - index of earthquake within origin file for which to get station list
    
    returns:
    repeat_subset - pandas dataframe that is a subset of the arrivals dataframe, containing only the stations which  have both a P- and S-wave pick for the given earthquake
    """
        
    # Get arrivals associated with this earthquake
    subset = assoc.loc[assoc['originid']==index]
    arrival_subset = arrivals.iloc[subset['arrivalid']-1]
    arrival_subset.reset_index(drop=True,inplace=True)

    # Get station names
    stations = arrival_subset['stationcode']

    # Get list of stations that have both P and S pick
    repeats = []
    for station in stations:
        sub = arrival_subset.loc[(arrival_subset['stationcode']==station)]
        if len(np.unique(sub['phase']))>1:
            repeats.append(station)
    repeat_subset = arrival_subset.loc[(arrival_subset['stationcode'].isin(repeats))]
    repeat_subset = repeat_subset.drop_duplicates(subset=['stationcode'])
    repeat_subset.reset_index(drop=True,inplace=True)

    return arrival_subset,repeat_subset 

def remove_compliance(st):
    """
    Removes compliance noise from the vertical channel using the pressure channel
    
    inputs:
    st - obsPy stream object containing a vertical and pressure channel
    
    outputs:
    zcorr - vertical trace data corrected for compliance
    """
    # Resample the pressure channel to 100 Hz:
    st.resample(100)

    # Make sure all traces are the same length:
    if (min(tr.stats.npts for tr in st)==max(tr.stats.npts for tr in st))==False:
        for tr in st:
            tr.data = tr.data[:min(tr.stats.npts for tr in st)]
    
    zchan = st.select(channel='*Z')
    pchan = st.select(channel='*H')
    n2 = zchan[0].stats.npts
    f = np.fft.rfftfreq(n2,1/zchan[0].stats.sampling_rate)
    ftZ = np.fft.fft(zchan[0].data,n=n2)
    ftP = np.fft.fft(pchan[0].data,n=n2)
    cZZ = np.abs(ftZ*np.conj(ftZ))
    cPP = np.abs(ftP * np.conj(ftP))
    cPZ = np.abs(np.conj(ftP)*ftZ)
    transPZ = cPZ / cPP
    zcorr_spec = ftZ - (ftP * transPZ)
    zcorr = np.fft.ifft(zcorr_spec)
    
    # Coherence:
    coh = ((cPZ)**2)/(cZZ * cPP)


    return zcorr,coh

def plot_fourier(st,zcorr):
    """
    Plots a 2x2 figure comparing the spectra of raw and compliance-corrected vertical channels
    
    Inputs:
    st = raw obspy stream object
    zcorr = vector of seismic data representing the corrected raw vertical data from st
    
    Outputs:
    fig = figure object 
    """
    
    %matplotlib inline
    fig, axs = plt.subplots(2, 2)
    print('station = '+st[0].stats.station+', '+'depth = '+str(elev))
    axs[0,0].specgram(st[3],Fs=100,vmin=0,vmax=100)
    axs[0,0].title.set_text('Raw Vertical Channel')
    axs[0,0].set_xlabel('Time(s)');
    axs[0,0].set_ylabel('Frequency (Hz)')
    axs[1,0].specgram(np.real(zcorr),Fs=100,vmin=0,vmax=100)
    axs[1,0].title.set_text('Compliance-corrected Vertical Channel')
    axs[1,0].set_xlabel('Time (s)')
    axs[1,0].set_ylabel('Frequency (Hz)')

    # Number of samplepoints
    N = 132000
    # sample spacing
    T = 1.0 / 100.0
    xf = np.linspace(int(0.0), int(1.0/(2.0*T)), int(N/2))


    y = st[3]
    yf = scipy.fftpack.fft(y)
    axs[0,1].plot(xf, 2.0/N * np.abs(yf[:N//2]))
    axs[0,1].title.set_text('Fourier Transform: Raw Vertical Channel')
    axs[0,1].set_xlabel('Frequency (Hz)');
    y = zcorr
    yf = scipy.fftpack.fft(y)
    axs[1,1].plot(xf, 2.0/N * np.abs(yf[:N//2]))
    axs[1,1].title.set_text('Fourier Transform: Compliance-corrected Vertical Channel')
    axs[1,1].set_xlabel('Frequency (Hz)');
    return fig
