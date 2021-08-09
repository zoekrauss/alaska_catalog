import pathlib
import obspy
from obspy.signal.cross_correlation import correlate
from obspy.signal.cross_correlation import xcorr_max
import numpy as np
import h5py
import matplotlib.pyplot as plt


def correlate_master(master_event,waveforms):

    # open file for output
    home_dir = str(pathlib.Path().absolute())
    out_file = h5py.File(home_dir + "/statla_correlations.h5","w")

    # make some arrays for storing output
    shifts = np.zeros((len(waveforms)))
    correlation_coefficients = np.zeros((len(waveforms)))

    # make a counter for percentage progress output
    p = 10
    
    for i in range(len(waveforms)):

        # correlate master event and waveform i
        correlation_timeseries = correlate(master_event,waveforms[i],master_event.stats.npts)
        shift, correlation_coefficient = xcorr_max(correlation_timeseries)

        # save output
        shifts[i] = shift
        correlation_coefficients[i] = correlation_coefficient

        # give the user progress output every 10% complete
        if i/len(waveforms)*100 > p:
            print("Correlated master event with " + str(p) + "% of events")
            p += 10
            
            
    # write output to file
    out_file.create_dataset("correlation_coefficients",data=correlation_coefficients)
    out_file.create_dataset("shifts",data=shifts)

    # close output file
    out_file.close()
    
    # give final output
    print("Correlated master event with 100% of events")
    return correlation_coefficients, shifts


def threshold_detection_plot(aligned_waves,correlation_coefficients,threshold_correlation_coefficients,shifts,threshold):
    
    # get dimensions of waveform matrix
    num_traces, trace_len = aligned_waves.shape
    # make subplots to plot results of correlation and thresholding
    fig,ax = plt.subplots(nrows=2,ncols=1,gridspec_kw={'height_ratios':[2,1]},figsize=(10,10))
    
    # plot aligned waveforms
    ax[0].imshow(aligned_waves,vmin=-0.5,vmax=0.5,aspect = 'auto',extent=[0,trace_len,num_traces,0],cmap='seismic')
    ax[0].set_xticks([0,trace_len/2,trace_len])
    ax[0].set_xticklabels(['0','250','500'])
    ax[0].set(ylabel="Time (seconds)")
    ax[0].set(xlabel="Event number")
    ax[0].axhline(y=len(threshold_correlation_coefficients),color='k',linestyle='dashed')
    ax[0].text(trace_len+100,len(threshold_correlation_coefficients)+15,"Threshold: " + str(threshold))
    ax[0].set_title("Aligned waveforms of detected events")

    # plot histogram of cross correlation coefficients
    ax[1].hist(abs(correlation_coefficients),20)
    ax[1].set(xlabel="Absolute normalized cross correlation coefficient")
    ax[1].set(ylabel="Number of events")
    ax[1].axvline(x=threshold,color='k',linestyle='dashed')
    ax[1].text(threshold+0.02,100,"Threshold: " + str(threshold))
    ax[1].set_title("Histogram of correlation coefficients")
    
    # fix subplot axes and show plot
    plt.tight_layout()
    plt.show()
    
    
def threshold_detections(waveforms,correlation_coefficients,shifts,threshold):
    
    # get trace length for later use
    trace_len = waveforms[0].stats.npts
    
    # get indices of sorted cross correlation coefficients
    sort_index = np.argsort(abs(correlation_coefficients))[::-1]

    # apply cross correlation threshold to waveforms and get matrix of aligned waveforms
    aligned_waves = np.zeros((len(waveforms),trace_len))
    count = 0
    for i in sort_index:

            # get trace from obspy stream
            trace = waveforms[i].data
            if shifts[i] > 0:
                aligned_trace = np.append(np.zeros(abs(int(shifts[i]))),trace)
                if len(aligned_trace) < trace_len:
                    aligned_trace = np.append(aligned_trace,np.zeros(trace_len-len(aligned_trace)))
                aligned_waves[count,:] = aligned_trace[:trace_len]
            else:
                aligned_trace = trace[abs(int(shifts[i])):]
                aligned_waves[count,:len(aligned_trace)] = aligned_trace
            
            count += 1
            
    # flip polarity of waves with negative cross correlation coefficients for plotting
    signs = np.sign(correlation_coefficients).reshape(len(correlation_coefficients),1)
    aligned_waves = np.multiply(aligned_waves,signs[sort_index])
            
    # normalize amplitude of waves for plotting
    aligned_waves = np.divide(aligned_waves,np.amax(np.abs(aligned_waves),axis=1,keepdims=True))
    
    # apply the threshold to vector of correlation coefficients 
    threshold_correlation_coefficients = correlation_coefficients[abs(correlation_coefficients) > threshold] 
    
    # report number of events to be made into templates
    print("\nThreshold of " + str(threshold) + " yields " + str(len(threshold_correlation_coefficients))  + " events.\n")
    
    # produce plots of aligned waves and histogram of correlation coefficients
    threshold_detection_plot(aligned_waves,correlation_coefficients,threshold_correlation_coefficients,shifts,threshold)
    

