import pandas as pd


#module imports
from pupilMetrics import pupilMetrics
from clustering import automatedSpaceClustering
from clustering import automatedSpaceTimeClustering
from fixationBasedPupilAnalysis import fixationBasedPupilAnalysis
from fineGrainedScanPath import fineGrainedScanPath
from saccadeCalculation import saccadeCalculation
from mLDataPreparation import mLDataPreparation

#log files
fixationLogFile = "../data/sample/fixationLog.csv"
fixationLogMetricsFile = "../out/fixationLog_metrics.csv"

# demo participant and demo task
participant = "P01"
task = "t7_eclipse (C_5.java)"

# lines with brackets and empty lines that should be skipped in the source-code
linestoIgnore = {"t7_eclipse (C_5.java)": [2,5,7,10,14,19,25,26,27,28,31,32,33,34]} 

#Specifiy the fragments coordinates within the source-code file e.g., C_5  (see code comprehension tasks/C_5.java)
#Format: {filename: {"fragmentname":"[lines range]"}}
persoAOI = {"t7_eclipse (C_5.java)": { 
                    "array list definition": [9,9],
                    "objects declaration": [11,13],
                    "add to array": [15,18],                    
                    "while loop": [20,25],  
                    "next function": [29,31],
                    "question": [37,37]
                    }}

#pupil parameters

#file with cleaned pupil data (following the pipeline proposed in Stefan Zugal, Jakob Pinggera, Manuel Neurauter, Thomas Maran, and Barbara 1270 Weber. 2017. Cheetah experimental platform web 1.0: cleaning pupillary data. 1271 arXiv preprint arXiv:1703.09468 (2017).)
pupilLogFile = "../data/sample/pupilLog.csv"
# file where pupil metrics will be saved 
PupilMetricsFile = "../out/PupilLog_metrics.csv"

# file with participants' highlights (we use the file with the intial highlights in this example)
highlightsSummaryfile = "../data/highlights/initial.csv";

#onset and offset of the time window to be used in the contextulization of fixations with resepct to the pupil signal
pupil_onset = 400 # in ms
pupil_offset = 1100 # in ms

#duration of the time window to be used in the contextualization of fixations with respect to the (sub) scan-paths
FineGrainedAttentionMap_window = 2000 #in ms


#clustering parameters to be used in the contextulization of fixations in space and time
longFixationThreshold = 250 # only the fixations with a duration (in ms) above this threshold will be used in the clustering
eps_val_AutomatedSpaceClustering = 3 #Space clustering: the maximum distance between two samples (in columns) for one to be considered as in the neighborhood of the other. https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
min_samples_val_AutomatedSpaceClustering= 4 #Sapce clustering: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself. https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
eps_val_AutomatedSpaceTimeClustering  = 800 # Space time clustering: the maximum distance between two samples (in time in ms) for one to be considered as in the neighborhood of the other. https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
min_samples_val_AutomatedSpaceTimeClustering = 2 #Space time clustering: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself. https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html


#File where the dataset prepared for Machine learning will be saved
outMLDataLines = "../out/sampleMLDataLines.csv" #this file will be used for predictions at the line level
outMLDataFragments = "../out/sampleMLDataFragments.csv" #this file will be used for predictions at the fragment level



#keep only lines of code with content (i.e., ignore the empty lines and the lines with brackets in the source-code from the calculation of features)
fixationLog = pd.read_csv(fixationLogFile)
fixationLog = fixationLog.loc[~fixationLog['source_file_line'].isin(linestoIgnore[task])]
fixationLog.to_csv(fixationLogMetricsFile)


#compute pupillary metrics: peaks and peaks amplitude
pupilMetrics(pupilLogFile,PupilMetricsFile)


#contextulization of fixations in space and time
automatedSpaceClustering(fixationLogMetricsFile,longFixationThreshold,eps_val_AutomatedSpaceClustering,min_samples_val_AutomatedSpaceClustering)
automatedSpaceTimeClustering(fixationLogMetricsFile,longFixationThreshold,eps_val_AutomatedSpaceTimeClustering,min_samples_val_AutomatedSpaceTimeClustering)


#contextualization of fixations with respect to the pupil signal 
fixationBasedPupilAnalysis(fixationLogMetricsFile,PupilMetricsFile,pupil_onset,pupil_offset)

#contextualization of fixations with respect to the (sub) scan-pathss
fineGrainedScanPath(fixationLogMetricsFile,FineGrainedAttentionMap_window,task)

#contextualization of fixations with respect to the preceding saccades
saccadeCalculation(fixationLogMetricsFile)

#generate ML datasets
mLDataPreparation(fixationLogMetricsFile,outMLDataLines,outMLDataFragments,task,persoAOI,highlightsSummaryfile,participant,linestoIgnore)

