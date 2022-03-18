import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np

def isinFLCluster(val,df,cl,line,attribute):
    searchSet = df[df['Timestamp']==val]
    if len(searchSet)==0:
        return cl
    else:
        return str(line)+";"+str(searchSet.iloc[0][attribute])

def isinCluster(filtredlog,val,attribute):
    searchSet = filtredlog[filtredlog['Timestamp']==val]
    if len(searchSet)==0:
        return ''
    else:
        return searchSet.iloc[0][attribute]

def isValid(val):
    if val=='' or '-1' in str(val):
        return 0
    return 1 


def automatedSpaceClustering(merged_imo_itrace_summerized,longFixationThreshold,eps_val_AutomatedSpaceClustering,min_samples_val_AutomatedSpaceClustering):

    log = pd.read_csv(merged_imo_itrace_summerized, index_col=0)

    # Keep only fixations above a given treshold
    filtredlog = log[log["Fixation Duration"]>=longFixationThreshold].copy(deep=True) 
    filtredlog['Spacecluster'] = ''

    for line in filtredlog["source_file_line"].unique():
        lineFiltredLog = filtredlog.loc[filtredlog["source_file_line"]==line].copy(deep=True) 
        # Spacial clustering with DBSCAN
        lineFiltredLog['Spacecluster'] = DBSCAN(eps = eps_val_AutomatedSpaceClustering, min_samples = min_samples_val_AutomatedSpaceClustering).fit_predict(lineFiltredLog[['source_file_col']])
        filtredlog['Spacecluster'] = filtredlog.apply(lambda x: isinFLCluster(x['Timestamp'], lineFiltredLog, x['Spacecluster'], line, 'Spacecluster'), axis = 1)


    log['Spacecluster'] = log.apply(lambda x: isinCluster(filtredlog,x['Timestamp'],'Spacecluster'), axis = 1)
    log['IsfixationInSpatialCluster'] = log['Spacecluster'].apply(isValid)


    log.to_csv(merged_imo_itrace_summerized)


def automatedSpaceTimeClustering(merged_imo_itrace_summerized,longFixationThreshold,eps_val_AutomatedSpaceTimeClustering,min_samples_val_AutomatedSpaceTimeClustering):

    log = pd.read_csv(merged_imo_itrace_summerized, index_col=0)


    # Keep only fixations above a given treshold
    filtredlog = log[log["Fixation Duration"]>=longFixationThreshold].copy(deep=True) 
    filtredlog['SpaceTimeCluster'] = ''


    for line in filtredlog["source_file_line"].unique():
        lineFiltredLog = filtredlog.loc[filtredlog["source_file_line"]==line].copy(deep=True)
        # Spacial clustering with DBSCAN
        lineFiltredLog['SpaceTimeCluster'] = DBSCAN(eps = eps_val_AutomatedSpaceTimeClustering, min_samples = min_samples_val_AutomatedSpaceTimeClustering).fit_predict(lineFiltredLog[['Timestamp']])
        filtredlog['SpaceTimeCluster'] = filtredlog.apply(lambda x: isinFLCluster(x['Timestamp'], lineFiltredLog, x['SpaceTimeCluster'], line, 'SpaceTimeCluster'), axis = 1)


    log['SpaceTimeCluster'] = log.apply(lambda x: isinCluster(filtredlog,x['Timestamp'],'SpaceTimeCluster'), axis = 1)
    log['IsfixationInSpatialTemporalCluster'] = log['SpaceTimeCluster'].apply(isValid)


    log.to_csv(merged_imo_itrace_summerized)



