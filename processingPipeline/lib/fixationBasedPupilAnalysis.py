import pandas as pd
import math , numpy as np


def computeCountPupilPeaks(val , pupil_onset , pupil_offset , pupilLog):
    startime = val+pupil_onset
    endtime = val+pupil_offset       
    dataPointsInRange = pupilLog[pupilLog['Timestamp'].between(startime, endtime)]
    value = len(dataPointsInRange[dataPointsInRange["is_peak"]==True])
    return value


def computePupilPeaksAmplitude(val,agg , pupil_onset , pupil_offset , pupilLog):
    startime = val+pupil_onset
    endtime = val+pupil_offset
    dataPointsInRange = pupilLog[pupilLog['Timestamp'].between(startime, endtime)]

    if agg=='max':
        value = dataPointsInRange[dataPointsInRange["is_peak"]==True]["amplitude"].max()
    if agg=='min':
        value = dataPointsInRange[dataPointsInRange["is_peak"]==True]["amplitude"].min()
    if agg=='median':
        value = dataPointsInRange[dataPointsInRange["is_peak"]==True]["amplitude"].median()
    if agg=='std':
        value = dataPointsInRange[dataPointsInRange["is_peak"]==True]["amplitude"].std()
    if agg=='mean':
        value = dataPointsInRange[dataPointsInRange["is_peak"]==True]["amplitude"].mean()  
    if agg=='sum':
        value = dataPointsInRange[dataPointsInRange["is_peak"]==True]["amplitude"].sum()  
    return value

def computePupilSize(val,agg , pupil_onset , pupil_offset , pupilLog):
    startime = val+pupil_onset
    endtime = val+pupil_offset
    dataPointsInRange = pupilLog[pupilLog['Timestamp'].between(startime, endtime)]
    if agg=='mean':
        value = dataPointsInRange["pupilSize"].mean()
    if agg=='median':
        value = dataPointsInRange["pupilSize"].median()
    if agg=='max':
        value = dataPointsInRange["pupilSize"].max()
    if agg=='min':
        value = dataPointsInRange["pupilSize"].min()
    if agg=='std':
        value = dataPointsInRange["pupilSize"].std()
    if agg=='sum':
        value = dataPointsInRange["pupilSize"].sum()          
    return value


def fixationBasedPupilAnalysis(merged_imo_itrace_summerized,pupilLogFile,pupil_onset,pupil_offset):



    fixationLog = pd.read_csv(merged_imo_itrace_summerized, index_col=0) 
    pupilLog = pd.read_csv(pupilLogFile)


    fixationLog['fixation_NumberOfPupilPeaks'] = fixationLog.apply(lambda x: computeCountPupilPeaks(x['Timestamp'] , pupil_onset , pupil_offset , pupilLog), axis = 1)

    fixationLog['fixation_PupilSize_mean'] = fixationLog.apply(lambda x: computePupilSize(x['Timestamp'], 'mean' , pupil_onset , pupil_offset , pupilLog), axis = 1)
    fixationLog['fixation_PupilSize_median'] = fixationLog.apply(lambda x: computePupilSize(x['Timestamp'], 'median' , pupil_onset , pupil_offset , pupilLog), axis = 1)
    fixationLog['fixation_PupilSize_max'] = fixationLog.apply(lambda x: computePupilSize(x['Timestamp'], 'max' , pupil_onset , pupil_offset , pupilLog), axis = 1)
    fixationLog['fixation_PupilSize_min'] = fixationLog.apply(lambda x: computePupilSize(x['Timestamp'], 'min' , pupil_onset , pupil_offset , pupilLog), axis = 1)
    fixationLog['fixation_PupilSize_std'] = fixationLog.apply(lambda x: computePupilSize(x['Timestamp'], 'std' , pupil_onset , pupil_offset , pupilLog), axis = 1)
    fixationLog['fixation_PupilSize_sum'] = fixationLog.apply(lambda x: computePupilSize(x['Timestamp'], 'sum' , pupil_onset , pupil_offset , pupilLog), axis = 1)

    fixationLog['fixation_PupilPeaksAmplitude_mean'] = fixationLog.apply(lambda x: computePupilPeaksAmplitude(x['Timestamp'], 'mean' , pupil_onset , pupil_offset , pupilLog), axis = 1)
    fixationLog['fixation_PupilPeaksAmplitude_median'] = fixationLog.apply(lambda x: computePupilPeaksAmplitude(x['Timestamp'], 'median' , pupil_onset , pupil_offset , pupilLog), axis = 1)
    fixationLog['fixation_PupilPeaksAmplitude_max'] = fixationLog.apply(lambda x: computePupilPeaksAmplitude(x['Timestamp'], 'max' , pupil_onset , pupil_offset , pupilLog), axis = 1)
    fixationLog['fixation_PupilPeaksAmplitude_min'] = fixationLog.apply(lambda x: computePupilPeaksAmplitude(x['Timestamp'], 'min' , pupil_onset , pupil_offset , pupilLog), axis = 1)
    fixationLog['fixation_PupilPeaksAmplitude_std'] = fixationLog.apply(lambda x: computePupilPeaksAmplitude(x['Timestamp'], 'std' , pupil_onset , pupil_offset , pupilLog), axis = 1)
    fixationLog['fixation_PupilPeaksAmplitude_sum'] = fixationLog.apply(lambda x: computePupilPeaksAmplitude(x['Timestamp'], 'sum' , pupil_onset , pupil_offset , pupilLog), axis = 1)


    fixationLog.to_csv(merged_imo_itrace_summerized) 




