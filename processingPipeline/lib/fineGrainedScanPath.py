import pandas as pd
import networkx as nx
import math , pywt , numpy as np
from scipy.spatial import distance
from scipy.stats import entropy



def computeContainingGraphMetric(val , weightsdf,col):
	selection = weightsdf.loc[(val >= weightsdf["start"]) & (val <= weightsdf["end"])]
	colValue = np.nan if len(selection)==0 else selection.iloc[0][col]
	return colValue


def fineGrainedScanPath(merged_imo_itrace_summerized,FineGrainedAttentionMap_window,task):

	#initiate an empty dataframe with the following columns
	weightsdf = pd.DataFrame(columns=['start', 'end', 'averageWeightedDegree','nTransitionsWeighted','nNodes','Density','adjacency_matrix_entropy','SourceStimuliName'])

	fixationLog = pd.read_csv(merged_imo_itrace_summerized, index_col=0) 

	#initial window: start=timestamp of the first fixation in the log; end=start+FineGrainedAttentionMap_window (i.e., definined as a parameter in main.py)
	start = fixationLog.iloc[0]["Timestamp"]
	end = start+FineGrainedAttentionMap_window
	
	ColumnContainingAOIFocus= 'source_file_line' 

	#iterate over the fixations in the fixatgionLog
	while start<fixationLog.iloc[-1]["Timestamp"]:

		#find all the fixations within the time window of which the bounds are specified by start and end
	    dataPointsInRange = fixationLog[fixationLog['Timestamp'].between(start, end)]
	    i = 0 
	    #counter of transitons
	    dictr = {}
	    while i<len(dataPointsInRange)-1:
	        
	    	#if there is a transitions (i.e., fixation on row x followed by a fixation on row y)
	        if dataPointsInRange.iloc[i][ColumnContainingAOIFocus]!=dataPointsInRange.iloc[i+1][ColumnContainingAOIFocus]: 

	        	# label the transition
	            transition = str(dataPointsInRange.iloc[i][ColumnContainingAOIFocus])+"-"+str(dataPointsInRange.iloc[i+1][ColumnContainingAOIFocus])

	            # if the transition already exists then increments the corresponding counter in dictr, otherwise, add the new transition to dictr
	            if transition in dictr.keys():
	                dictr[transition][0] = dictr[transition][0]+1
	            else:
	                
	                dictr[transition] = [1,[
	                str(dataPointsInRange.iloc[i][ColumnContainingAOIFocus]),
	                str(dataPointsInRange.iloc[i+1][ColumnContainingAOIFocus])
	                ]]

	        i = i+1

	    #create a graph with the transitions in dictr. The weights of the edges correspond to their correspond counter value in dictr
	    G=nx.MultiDiGraph()
	    for key in dictr:
	        G.add_edge(dictr[key][1][0], dictr[key][1][1], weight = dictr[key][0])


	    for edge in G.edges(data=True): edge[2]['label'] = edge[2]['weight']
	    node_label = nx.get_node_attributes(G,'id')
	    pos = nx.spring_layout(G)
	    node_label = nx.get_node_attributes(G,'id')
	    pos = nx.spring_layout(G)

	    sumWeightedDegree = 0;
	    wds = dict(G.degree(weight='weight'))

	    r=0
	    for d in wds:
	        r = r+1
	        sumWeightedDegree = sumWeightedDegree + wds[d]

	    #compute average weighted degree of the graph 
	    averageWeightedDegree = float(sumWeightedDegree/(r)) if r>0 else 0 #for plotting purpose np.nan. changed to 0
	   

	    transitionsCountWeighted = 0

	    for u,v,a in G.edges(data=True):

	       transitionsCountWeighted = transitionsCountWeighted + a['weight']

	    #compute the number of nodes in the graph
	    nNodes = G.number_of_nodes()
	    #compute the density of the graph
	    density = float(transitionsCountWeighted/nNodes) if nNodes>0 else np.nan
	    
	    #compute the entropy of the adjacency_matrix_entropy matrix of the graph
	    adjacency_matrix_entropy = np.nan
	    if(len(list(G.nodes))>0):
	        adjacency_matrix_entropy = entropy(nx.to_numpy_array(G).flatten(), base=2)


	    weightsdf = weightsdf.append({'start': start, 'end': end, 'averageWeightedDegree': averageWeightedDegree, 'nTransitionsWeighted': transitionsCountWeighted, 'nNodes': nNodes, 'Density':density, 'adjacency_matrix_entropy':adjacency_matrix_entropy, 'SourceStimuliName':task}, ignore_index=True)


	    start = end
	    end = start+FineGrainedAttentionMap_window

	#compute graph metrics for the contextualization of fixations with respect to the (sub) scan-paths
	
	fixationLog["Sub-scanNtranstions"] = fixationLog.apply(lambda x: computeContainingGraphMetric(x["Timestamp"],weightsdf,'nTransitionsWeighted'), axis = 1)
	fixationLog["Sub-scanNUniqueVisits"] = fixationLog.apply(lambda x: computeContainingGraphMetric(x["Timestamp"],weightsdf,'nNodes'), axis = 1)
	fixationLog["Sub-scanDensity"] = fixationLog.apply(lambda x: computeContainingGraphMetric(x["Timestamp"],weightsdf,'Density'), axis = 1)
	fixationLog["Sub-scanAverageWeightedDegree"] = fixationLog.apply(lambda x: computeContainingGraphMetric(x["Timestamp"],weightsdf,'averageWeightedDegree'), axis = 1)
	fixationLog["Sub-scanEntropy"] = fixationLog.apply(lambda x: computeContainingGraphMetric(x["Timestamp"],weightsdf,'adjacency_matrix_entropy'), axis = 1)


	fixationLog.to_csv(merged_imo_itrace_summerized)


