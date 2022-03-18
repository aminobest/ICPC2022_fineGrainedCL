import pandas as pd
from scipy.signal import find_peaks, peak_prominences
import numpy as np


def ispeak(val,peaks):       
    return True if val in peaks else False

def findamplitude(val,peaks,prominences):       
    if val in peaks:
        index_val = np.where(peaks == val) 
        return prominences[index_val][0]
    else:
        return np.nan

def pupilMetrics(pupilLogFile,outFile):

	df = pd.read_csv(pupilLogFile)
	peaks = find_peaks(df['pupilSize'])[0]
	prominences = peak_prominences(df['pupilSize'], peaks)[0]

	# find the peaks and calculate their amplitude
	df['index1'] = df.index
	df["is_peak"] = df.apply(lambda x:ispeak(x['index1'],peaks) , axis=1)
	df["amplitude"] = df.apply(lambda x:findamplitude(x['index1'],peaks,prominences) , axis=1)

	df[['Timestamp','ET_PupilLeft', 'ET_PupilRight','pupilSize','SourceStimuliName','is_peak','amplitude']].to_csv(outFile)

