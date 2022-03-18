import pandas as pd
from scipy.spatial import distance



def calculateDirectiontoPreviousPoint(index,direction,log):
	if index!=0:
	    
	    currentFixDuration = log.iloc[index]["Fixation Duration"]
	    prevFixDuration = log.iloc[index-1]["Fixation Duration"]
	     
	    prevcol = log.iloc[index-1]["source_file_col"]
	    prevrow = log.iloc[index-1]["source_file_line"]

	    currentcol = log.iloc[index]["source_file_col"]
	    currentrow = log.iloc[index]["source_file_line"]

	    if prevrow!=currentrow and direction=='diagonal':
	        return 1

	    elif prevcol!=currentcol and prevrow==currentrow  and direction=='horizental':
	        return 1

	    elif prevcol==currentcol and prevrow==currentrow and direction=='no':
	        return 1

	return 0

def calculateAmplitudetoPreviousPoint(index,log):
	if index!=0:
	    
	    currentFixDuration = log.iloc[index]["Fixation Duration"]
	    prevFixDuration = log.iloc[index-1]["Fixation Duration"]
	     
	    prevcol = log.iloc[index-1]["source_file_col"]
	    prevrow = log.iloc[index-1]["source_file_line"]

	    currentcol = log.iloc[index]["source_file_col"]
	    currentrow = log.iloc[index]["source_file_line"]

	    return distance.euclidean([prevcol, prevrow], [currentcol, currentrow])

	return 0



def saccadeCalculation(merged_imo_itrace_summerized):


	log = pd.read_csv(merged_imo_itrace_summerized, index_col=0) 


	log["index"] = log.index

	log["non-horizontal saccade"] = log.apply(lambda x: calculateDirectiontoPreviousPoint(x["index"],'diagonal',log) ,axis=1)
	log["horizontal saccade"] = log.apply(lambda x: calculateDirectiontoPreviousPoint(x["index"],'horizental',log) ,axis=1)
	log["no saccade"] = log.apply(lambda x: calculateDirectiontoPreviousPoint(x["index"],'no',log) ,axis=1)
	log["saccadic amplitude"] = log.apply(lambda x: calculateAmplitudetoPreviousPoint(x["index"],log) ,axis=1)
	


	log.to_csv(merged_imo_itrace_summerized) 


