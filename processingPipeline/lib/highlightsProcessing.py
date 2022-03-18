import pandas as pd
import re


# highlightsSummaryfile = "../data/highlights/initial.csv";
# participant = "P01"
# task = "t7_eclipse (C_5.java)"

# linestoIgnore = {"t7_eclipse (C_5.java)": [2,5,7,10,14,19,25,26,27,28,31,32,33,34]} #ignore lines with brackets and empty lines
# persoAOI = {"t7_eclipse (C_5.java)": {
#                     "array list definition": [9,9],
#                     "objects declaration": [11,13],
#                     "add to array": [15,18],                    
#                     "while loop": [20,25],  
#                     "next function": [29,31],
#                     "question": [37,37]
#                     }}

def getFragment(line,task,persoAOI):

        for aoi in persoAOI[task]:

            lineInterval = range(persoAOI[task][aoi][0],persoAOI[task][aoi][1]+1)
            if line in lineInterval:
               
                return aoi
        return 'other'


def highlightsProcessing(highlightsSummaryfile,participant,task,persoAOI,linestoIgnore):

	c_eclipseCode = re.search('\\((.+?).java', task).group(1);

	highlights = pd.read_csv(highlightsSummaryfile)
	highlights = highlights[highlights["participant"]==participant]
	highlights = highlights[highlights["code"]==c_eclipseCode]
	highlightedLines = highlights["line"].unique().tolist()

	# ignore empty lines and lines with brackets
	highlightedLines = list(set(highlightedLines).difference(linestoIgnore[task]))

	highlightedCorespondingFragements = set([getFragment(i,task,persoAOI) for i in highlightedLines])

	

	return highlightedLines, highlightedCorespondingFragements



# highlightsProcessing(highlightsSummaryfile,participant,task)