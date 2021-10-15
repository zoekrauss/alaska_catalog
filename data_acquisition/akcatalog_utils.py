import obspy
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import pandas as pd
import time; import datetime
from obspy.core.utcdatetime import UTCDateTime
from obspy.clients.fdsn.client import Client
import pyasdf

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

def calc_snr(stream,sampleind,phase):
    """
    # Calculate SNR of arrival
    # INPUTS:
    # stream = obspy-formatted waveform object
    # sampleind = index in the stream's data of desired arrival for which to calculate SNR
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
        data = stream[0].data
        sr = int(stream[0].stats.sampling_rate)
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
    for trace in st:
        if trace.stats.npts != 132000:
            trace.trim(starttime = trace.stats.starttime,endtime = trace.stats.endtime-(.01*(trace.stats.npts-132000)))

    n2 = st[3].stats.npts
    f = np.fft.rfftfreq(n2,1/st[3].stats.sampling_rate)
    ftZ = np.fft.fft(st[3].data,n=n2)
    ftP = np.fft.fft(st[0].data,n=n2)
    cZZ = np.abs(ftZ*np.conj(ftZ))
    cPP = np.abs(ftP * np.conj(ftP))
    cPZ = np.abs(np.conj(ftP)*ftZ)
    transPZ = cPZ / cPP
    zcorr_spec = ftZ - (ftP * transPZ)
    zcorr = np.fft.ifft(zcorr_spec)


    return zcorr 