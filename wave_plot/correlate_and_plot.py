import obspy
import obspyh5
from correlation.master_event_correlation import correlate_master
from correlation.master_event_correlation import threshold_detections

# read waveforms
waveforms = obspy.read('waveforms.h5')

# filter waveforms
freq = [0.05,1]
waveforms.taper(max_percentage=0.1, max_length=30.)
waveforms.filter("bandpass",freqmin=freq[0],freqmax=freq[1])

# get vertical only waves
z_waveforms = waveforms.select(component="Z")

# set master event for correlation after plotting to see if it looks dispersive and has high SNR
master_idx = 567
master_event = z_waveforms[master_idx]
master_event.plot()

# set cross correlation threshold
threshold = 0.7

# correlate master with all other waveforms
correlation_coefficients, shifts = correlate_master(master_event,z_waveforms)

# apply threshold to choose best templates and make plots
threshold_detections(z_waveforms,correlation_coefficients,shifts,threshold)
