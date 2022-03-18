import pandas as pd
import numpy as np
import highlightsProcessing as hp


def countUniqueClusters(vals):
    return len(set([x for x in vals if str(x) != 'nan' and '-1' not in str(x)]))


def islineHighlighted(val,highlightedLines):
    return 1 if val[0] in highlightedLines else 0

def isFragmentHighlighted(val,highlightedCorespondingFragements):
    return 1 if val[0] in set(highlightedCorespondingFragements) else 0    


def mLDataPreparation(merged_imo_itrace_summerized,outMLDataLines,outMLDataFragments,task,persoAOI,highlightsSummaryfile,participant,linestoIgnore):

    log = pd.read_csv(merged_imo_itrace_summerized) 


    log["Fixation Duration 250"] = np.where(log["Fixation Duration"]>=250,log["Fixation Duration"],np.nan)
    log["Fixation Duration 500"] = np.where(log["Fixation Duration"]>=500,log["Fixation Duration"],np.nan)

    log["Fixation Count 250"] = np.where(log["Fixation Duration"]>=250,1,0)
    log["Fixation Count 500"] = np.where(log["Fixation Duration"]>=500,1,0)

    log["Fixation Count"] = 1


    log["fragment"] = log.apply(lambda x: hp.getFragment(x["source_file_line"],task,persoAOI), axis = 1)

    group_aggs = {                                 
                                                        "Fixation Duration": ['mean','sum','max','min','median','std'],
                                                        "Fixation Duration 250": ['mean','sum','max','min','median','std'],
                                                        "Fixation Duration 500": ['mean','sum','max','min','median','std'],

                                                        "Fixation Count": ['mean','sum','max','min','median','std'],
                                                        "Fixation Count 250": ['mean','sum','max','min','median','std'],
                                                        "Fixation Count 500": ['mean','sum','max','min','median','std'],

                                                        'fixation_NumberOfPupilPeaks' : ['mean','sum','max','min','median','std'],
                                                        'fixation_PupilSize_mean' : ['mean','sum','max','min','median','std'],
                                                        'fixation_PupilSize_median' : ['mean','sum','max','min','median','std'],
                                                        'fixation_PupilSize_max' : ['mean','sum','max','min','median','std'],
                                                        'fixation_PupilSize_min' : ['mean','sum','max','min','median','std'],
                                                        'fixation_PupilSize_std' : ['mean','sum','max','min','median','std'],
                                                        'fixation_PupilSize_sum' : ['mean','sum','max','min','median','std'],    
                                                        'fixation_PupilPeaksAmplitude_mean' : ['mean','sum','max','min','median','std'], 
                                                        'fixation_PupilPeaksAmplitude_median' : ['mean','sum','max','min','median','std'],
                                                        'fixation_PupilPeaksAmplitude_max' : ['mean','sum','max','min','median','std'],
                                                        'fixation_PupilPeaksAmplitude_min' : ['mean','sum','max','min','median','std'],
                                                        'fixation_PupilPeaksAmplitude_std' : ['mean','sum','max','min','median','std'],
                                                        'fixation_PupilPeaksAmplitude_sum' : ['mean','sum','max','min','median','std'],

                                                        "IsfixationInSpatialCluster": ['mean','sum','max','min','median','std'], 
                                                        "IsfixationInSpatialTemporalCluster": ['mean','sum','max','min','median','std'], 


                                                        "Spacecluster": [countUniqueClusters], #number of space clusters
                                                        "SpaceTimeCluster": [countUniqueClusters], #number of spacetime clusters

                                                         
                                                        "non-horizontal saccade": ['mean','sum','max','min','median','std'],
                                                        "horizontal saccade": ['mean','sum','max','min','median','std'],
                                                        "no saccade": ['mean','sum','max','min','median','std'],
                                                        "saccadic amplitude": ['mean','sum','max','min','median','std'],


                                                        "Sub-scanAverageWeightedDegree": ['mean','sum','max','min','median','std'],  
                                                        "Sub-scanNtranstions": ['mean','sum','max','min','median','std'],
                                                        "Sub-scanNUniqueVisits": ['mean','sum','max','min','median','std'],
                                                        "Sub-scanDensity": ['mean','sum','max','min','median','std'],
                                                        "Sub-scanEntropy": ['mean','sum','max','min','median','std'],

                                                     }



    highlightedLines, highlightedCorespondingFragements = hp.highlightsProcessing(highlightsSummaryfile,participant,task,persoAOI,linestoIgnore)



    linesgroup = log.groupby('source_file_line').agg(group_aggs)
    linesgroup["index"] = linesgroup.index
    linesgroup["label_LineHighlighted"] = linesgroup.apply(lambda x: islineHighlighted(x["index"],highlightedLines), axis=1) 

    linesgroup.columns = ['_'.join(col) for col in linesgroup.columns.values]

    linesgroup["fragment"] = linesgroup.apply(lambda x: hp.getFragment(x["index_"],task,persoAOI), axis=1) 
    linesgroup["participant"] = participant
    linesgroup["task"] = task



    fragmentgroup = log.groupby('fragment').agg(group_aggs)
    fragmentgroup["index"] = fragmentgroup.index
    fragmentgroup["label_fragmentHighlighted"] = fragmentgroup.apply(lambda x: isFragmentHighlighted(x["index"],highlightedCorespondingFragements), axis=1) 

    fragmentgroup.columns = ['_'.join(col) for col in fragmentgroup.columns.values]

    fragmentgroup["participant"] = participant
    fragmentgroup["task"] = task


    linesgroup[linesgroup.columns.drop('index_')].to_csv(outMLDataLines) 
    fragmentgroup[fragmentgroup.columns.drop('index_')].to_csv(outMLDataFragments) 
