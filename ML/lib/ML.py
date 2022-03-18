from numpy import mean
from numpy import std
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from imblearn.pipeline import make_pipeline
from imblearn import FunctionSampler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import os
from sklearn.ensemble import VotingClassifier
import sys
from numpy import savetxt
from random import sample

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from joblib import Parallel, delayed


#parameters

#features for training models to make predictions at the line level 
featuresLinePred = [
'Fixation Duration_mean',
'Fixation Duration_sum',
'Fixation Duration_max',
'Fixation Duration_min',
'Fixation Duration_median',
'Fixation Duration_std',
'Fixation Duration 250_mean',
'Fixation Duration 250_sum',
'Fixation Duration 250_max',
'Fixation Duration 250_min',
'Fixation Duration 250_median',
'Fixation Duration 250_std',
'Fixation Duration 500_mean',
'Fixation Duration 500_sum',
'Fixation Duration 500_max',
'Fixation Duration 500_min',
'Fixation Duration 500_median',
'Fixation Duration 500_std',
'Fixation Count_mean',
'Fixation Count_sum',
'Fixation Count_max',
'Fixation Count_min',
'Fixation Count_median',
'Fixation Count_std',
'Fixation Count 250_mean',
'Fixation Count 250_sum',
'Fixation Count 250_max',
'Fixation Count 250_min',
'Fixation Count 250_median',
'Fixation Count 250_std',
'Fixation Count 500_mean',
'Fixation Count 500_sum',
'Fixation Count 500_max',
'Fixation Count 500_min',
'Fixation Count 500_median',
'Fixation Count 500_std',
'fixation_NumberOfPupilPeaks_mean',
'fixation_NumberOfPupilPeaks_sum',
'fixation_NumberOfPupilPeaks_max',
'fixation_NumberOfPupilPeaks_min',
'fixation_NumberOfPupilPeaks_median',
'fixation_NumberOfPupilPeaks_std',
'fixation_PupilSize_mean_mean',
'fixation_PupilSize_mean_sum',
'fixation_PupilSize_mean_max',
'fixation_PupilSize_mean_min',
'fixation_PupilSize_mean_median',
'fixation_PupilSize_mean_std',
'fixation_PupilSize_median_mean',
'fixation_PupilSize_median_sum',
'fixation_PupilSize_median_max',
'fixation_PupilSize_median_min',
'fixation_PupilSize_median_median',
'fixation_PupilSize_median_std',
'fixation_PupilSize_max_mean',
'fixation_PupilSize_max_sum',
'fixation_PupilSize_max_max',
'fixation_PupilSize_max_min',
'fixation_PupilSize_max_median',
'fixation_PupilSize_max_std',
'fixation_PupilSize_min_mean',
'fixation_PupilSize_min_sum',
'fixation_PupilSize_min_max',
'fixation_PupilSize_min_min',
'fixation_PupilSize_min_median',
'fixation_PupilSize_min_std',
'fixation_PupilSize_std_mean',
'fixation_PupilSize_std_sum',
'fixation_PupilSize_std_max',
'fixation_PupilSize_std_min',
'fixation_PupilSize_std_median',
'fixation_PupilSize_std_std',
'fixation_PupilSize_sum_mean',
'fixation_PupilSize_sum_sum',
'fixation_PupilSize_sum_max',
'fixation_PupilSize_sum_min',
'fixation_PupilSize_sum_median',
'fixation_PupilSize_sum_std',
'fixation_PupilPeaksAmplitude_mean_mean',
'fixation_PupilPeaksAmplitude_mean_sum',
'fixation_PupilPeaksAmplitude_mean_max',
'fixation_PupilPeaksAmplitude_mean_min',
'fixation_PupilPeaksAmplitude_mean_median',
'fixation_PupilPeaksAmplitude_mean_std',
'fixation_PupilPeaksAmplitude_median_mean',
'fixation_PupilPeaksAmplitude_median_sum',
'fixation_PupilPeaksAmplitude_median_max',
'fixation_PupilPeaksAmplitude_median_min',
'fixation_PupilPeaksAmplitude_median_median',
'fixation_PupilPeaksAmplitude_median_std',
'fixation_PupilPeaksAmplitude_max_mean',
'fixation_PupilPeaksAmplitude_max_sum',
'fixation_PupilPeaksAmplitude_max_max',
'fixation_PupilPeaksAmplitude_max_min',
'fixation_PupilPeaksAmplitude_max_median',
'fixation_PupilPeaksAmplitude_max_std',
'fixation_PupilPeaksAmplitude_min_mean',
'fixation_PupilPeaksAmplitude_min_sum',
'fixation_PupilPeaksAmplitude_min_max',
'fixation_PupilPeaksAmplitude_min_min',
'fixation_PupilPeaksAmplitude_min_median',
'fixation_PupilPeaksAmplitude_min_std',
'fixation_PupilPeaksAmplitude_std_mean',
'fixation_PupilPeaksAmplitude_std_sum',
'fixation_PupilPeaksAmplitude_std_max',
'fixation_PupilPeaksAmplitude_std_min',
'fixation_PupilPeaksAmplitude_std_median',
'fixation_PupilPeaksAmplitude_std_std',
'fixation_PupilPeaksAmplitude_sum_mean',
'fixation_PupilPeaksAmplitude_sum_sum',
'fixation_PupilPeaksAmplitude_sum_max',
'fixation_PupilPeaksAmplitude_sum_min',
'fixation_PupilPeaksAmplitude_sum_median',
'fixation_PupilPeaksAmplitude_sum_std',
'IsfixationInSpatialCluster_mean',
'IsfixationInSpatialCluster_sum',
'IsfixationInSpatialCluster_max',
'IsfixationInSpatialCluster_min',
'IsfixationInSpatialCluster_median',
'IsfixationInSpatialCluster_std',
'IsfixationInSpatialTemporalCluster_mean',
'IsfixationInSpatialTemporalCluster_sum',
'IsfixationInSpatialTemporalCluster_max',
'IsfixationInSpatialTemporalCluster_min',
'IsfixationInSpatialTemporalCluster_median',
'IsfixationInSpatialTemporalCluster_std',
'Spacecluster_countUniqueClusters',
'SpaceTimeCluster_countUniqueClusters',
'non-horizontal saccade_mean',
'non-horizontal saccade_sum',
'non-horizontal saccade_max',
'non-horizontal saccade_min',
'non-horizontal saccade_median',
'non-horizontal saccade_std',
'horizontal saccade_mean',
'horizontal saccade_sum',
'horizontal saccade_max',
'horizontal saccade_min',
'horizontal saccade_median',
'horizontal saccade_std',
'no saccade_mean',
'no saccade_sum',
'no saccade_max',
'no saccade_min',
'no saccade_median',
'no saccade_std',
'saccadic amplitude_mean',
'saccadic amplitude_sum',
'saccadic amplitude_max',
'saccadic amplitude_min',
'saccadic amplitude_median',
'saccadic amplitude_std',
'Sub-scanAverageWeightedDegree_mean',
'Sub-scanAverageWeightedDegree_sum',
'Sub-scanAverageWeightedDegree_max',
'Sub-scanAverageWeightedDegree_min',
'Sub-scanAverageWeightedDegree_median',
'Sub-scanAverageWeightedDegree_std',
'Sub-scanNtranstions_mean',
'Sub-scanNtranstions_sum',
'Sub-scanNtranstions_max',
'Sub-scanNtranstions_min',
'Sub-scanNtranstions_median',
'Sub-scanNtranstions_std',
'Sub-scanNUniqueVisits_mean',
'Sub-scanNUniqueVisits_sum',
'Sub-scanNUniqueVisits_max',
'Sub-scanNUniqueVisits_min',
'Sub-scanNUniqueVisits_median',
'Sub-scanNUniqueVisits_std',
'Sub-scanDensity_mean',
'Sub-scanDensity_sum',
'Sub-scanDensity_max',
'Sub-scanDensity_min',
'Sub-scanDensity_median',
'Sub-scanDensity_std',
'Sub-scanEntropy_mean',
'Sub-scanEntropy_sum',
'Sub-scanEntropy_max',
'Sub-scanEntropy_min',
'Sub-scanEntropy_median',
'Sub-scanEntropy_std']


#features for training models to make predictions at the fragment level 
featuresFragmentPred = [
'Fixation Duration_mean',
'Fixation Duration_sum',
'Fixation Duration_max',
'Fixation Duration_min',
'Fixation Duration_median',
'Fixation Duration_std',
'Fixation Duration 250_mean',
'Fixation Duration 250_sum',
'Fixation Duration 250_max',
'Fixation Duration 250_min',
'Fixation Duration 250_median',
'Fixation Duration 250_std',
'Fixation Duration 500_mean',
'Fixation Duration 500_sum',
'Fixation Duration 500_max',
'Fixation Duration 500_min',
'Fixation Duration 500_median',
'Fixation Duration 500_std',
'Fixation Count_mean',
'Fixation Count_sum',
'Fixation Count_max',
'Fixation Count_min',
'Fixation Count_median',
'Fixation Count_std',
'Fixation Count 250_mean',
'Fixation Count 250_sum',
'Fixation Count 250_max',
'Fixation Count 250_min',
'Fixation Count 250_median',
'Fixation Count 250_std',
'Fixation Count 500_mean',
'Fixation Count 500_sum',
'Fixation Count 500_max',
'Fixation Count 500_min',
'Fixation Count 500_median',
'Fixation Count 500_std',
'fixation_NumberOfPupilPeaks_mean',
'fixation_NumberOfPupilPeaks_sum',
'fixation_NumberOfPupilPeaks_max',
'fixation_NumberOfPupilPeaks_min',
'fixation_NumberOfPupilPeaks_median',
'fixation_NumberOfPupilPeaks_std',
'fixation_PupilSize_mean_mean',
'fixation_PupilSize_mean_sum',
'fixation_PupilSize_mean_max',
'fixation_PupilSize_mean_min',
'fixation_PupilSize_mean_median',
'fixation_PupilSize_mean_std',
'fixation_PupilSize_median_mean',
'fixation_PupilSize_median_sum',
'fixation_PupilSize_median_max',
'fixation_PupilSize_median_min',
'fixation_PupilSize_median_median',
'fixation_PupilSize_median_std',
'fixation_PupilSize_max_mean',
'fixation_PupilSize_max_sum',
'fixation_PupilSize_max_max',
'fixation_PupilSize_max_min',
'fixation_PupilSize_max_median',
'fixation_PupilSize_max_std',
'fixation_PupilSize_min_mean',
'fixation_PupilSize_min_sum',
'fixation_PupilSize_min_max',
'fixation_PupilSize_min_min',
'fixation_PupilSize_min_median',
'fixation_PupilSize_min_std',
'fixation_PupilSize_std_mean',
'fixation_PupilSize_std_sum',
'fixation_PupilSize_std_max',
'fixation_PupilSize_std_min',
'fixation_PupilSize_std_median',
'fixation_PupilSize_std_std',
'fixation_PupilSize_sum_mean',
'fixation_PupilSize_sum_sum',
'fixation_PupilSize_sum_max',
'fixation_PupilSize_sum_min',
'fixation_PupilSize_sum_median',
'fixation_PupilSize_sum_std',
'fixation_PupilPeaksAmplitude_mean_mean',
'fixation_PupilPeaksAmplitude_mean_sum',
'fixation_PupilPeaksAmplitude_mean_max',
'fixation_PupilPeaksAmplitude_mean_min',
'fixation_PupilPeaksAmplitude_mean_median',
'fixation_PupilPeaksAmplitude_mean_std',
'fixation_PupilPeaksAmplitude_median_mean',
'fixation_PupilPeaksAmplitude_median_sum',
'fixation_PupilPeaksAmplitude_median_max',
'fixation_PupilPeaksAmplitude_median_min',
'fixation_PupilPeaksAmplitude_median_median',
'fixation_PupilPeaksAmplitude_median_std',
'fixation_PupilPeaksAmplitude_max_mean',
'fixation_PupilPeaksAmplitude_max_sum',
'fixation_PupilPeaksAmplitude_max_max',
'fixation_PupilPeaksAmplitude_max_min',
'fixation_PupilPeaksAmplitude_max_median',
'fixation_PupilPeaksAmplitude_max_std',
'fixation_PupilPeaksAmplitude_min_mean',
'fixation_PupilPeaksAmplitude_min_sum',
'fixation_PupilPeaksAmplitude_min_max',
'fixation_PupilPeaksAmplitude_min_min',
'fixation_PupilPeaksAmplitude_min_median',
'fixation_PupilPeaksAmplitude_min_std',
'fixation_PupilPeaksAmplitude_std_mean',
'fixation_PupilPeaksAmplitude_std_sum',
'fixation_PupilPeaksAmplitude_std_max',
'fixation_PupilPeaksAmplitude_std_min',
'fixation_PupilPeaksAmplitude_std_median',
'fixation_PupilPeaksAmplitude_std_std',
'fixation_PupilPeaksAmplitude_sum_mean',
'fixation_PupilPeaksAmplitude_sum_sum',
'fixation_PupilPeaksAmplitude_sum_max',
'fixation_PupilPeaksAmplitude_sum_min',
'fixation_PupilPeaksAmplitude_sum_median',
'fixation_PupilPeaksAmplitude_sum_std',
'IsfixationInSpatialCluster_mean',
'IsfixationInSpatialCluster_sum',
'IsfixationInSpatialCluster_max',
'IsfixationInSpatialCluster_min',
'IsfixationInSpatialCluster_median',
'IsfixationInSpatialCluster_std',
'IsfixationInSpatialTemporalCluster_mean',
'IsfixationInSpatialTemporalCluster_sum',
'IsfixationInSpatialTemporalCluster_max',
'IsfixationInSpatialTemporalCluster_min',
'IsfixationInSpatialTemporalCluster_median',
'IsfixationInSpatialTemporalCluster_std',
'Spacecluster_countUniqueClusters',
'SpaceTimeCluster_countUniqueClusters',
'non-horizontal saccade_mean',
'non-horizontal saccade_sum',
'non-horizontal saccade_max',
'non-horizontal saccade_min',
'non-horizontal saccade_median',
'non-horizontal saccade_std',
'horizontal saccade_mean',
'horizontal saccade_sum',
'horizontal saccade_max',
'horizontal saccade_min',
'horizontal saccade_median',
'horizontal saccade_std',
'no saccade_mean',
'no saccade_sum',
'no saccade_max',
'no saccade_min',
'no saccade_median',
'no saccade_std',
'saccadic amplitude_mean',
'saccadic amplitude_sum',
'saccadic amplitude_max',
'saccadic amplitude_min',
'saccadic amplitude_median',
'saccadic amplitude_std',
'Sub-scanAverageWeightedDegree_mean',
'Sub-scanAverageWeightedDegree_sum',
'Sub-scanAverageWeightedDegree_max',
'Sub-scanAverageWeightedDegree_min',
'Sub-scanAverageWeightedDegree_median',
'Sub-scanAverageWeightedDegree_std',
'Sub-scanNtranstions_mean',
'Sub-scanNtranstions_sum',
'Sub-scanNtranstions_max',
'Sub-scanNtranstions_min',
'Sub-scanNtranstions_median',
'Sub-scanNtranstions_std',
'Sub-scanNUniqueVisits_mean',
'Sub-scanNUniqueVisits_sum',
'Sub-scanNUniqueVisits_max',
'Sub-scanNUniqueVisits_min',
'Sub-scanNUniqueVisits_median',
'Sub-scanNUniqueVisits_std',
'Sub-scanDensity_mean',
'Sub-scanDensity_sum',
'Sub-scanDensity_max',
'Sub-scanDensity_min',
'Sub-scanDensity_median',
'Sub-scanDensity_std',
'Sub-scanEntropy_mean',
'Sub-scanEntropy_sum',
'Sub-scanEntropy_max',
'Sub-scanEntropy_min',
'Sub-scanEntropy_median',
'Sub-scanEntropy_std']


#type of cross-validation
prediction = "participant" #choose from: participant, task, participant-task 

#level of prediction
level  = "Lines"  #choose from: Lines, Fragments

#highlights used to provide labels
highlights = "initial" #choose from: "initial" , "revised"

#path of the highlights file used to provide labels
mldataset = "../datasets/"+highlights+"/MLData"+level+".csv"

#path where the prediction results will be stored
PredictionResultsfolder = "../out/"+highlights+"/"


#types of specific attributes in the dataframe
special_dtypes = {"participant": 'string', 'task': 'string'}

participantsToExclude = ["20.13"] #participants with bad data quality

# number of sampling iterations
numberOfSamplingIterations = 10

#number of features to select by the k best feature selection algorithm 
kbest = 20

#sampling approach: undersampling
sampling = 'low'

np.set_printoptions(threshold=np.inf)
features_importance = pd.DataFrame() 
f1 = list()
accuracy = list()
precision = list()
recall = list()


#functions

#models training and validation
def ml(iterations,dataset,features,label,equalizationMethod,evalMethod,level):

    i = 0

    classifier_f1 = []
    classifier_recall = []
    classifier_accuracy = []
    classifier_precision = []


    participantList = [val for val in dataset["participant"].unique()]
    taskList = [val for val in dataset["task"].unique()]

    feature_importances = pd.DataFrame() 

    dataset.fillna(0, inplace=True)


    balancedSamples = {}

    #for each iteration of the imbalanced classification
    while i < iterations:
        print("sampling iteration: "+str(i+1)+"/"+str(iterations))
        #conduct the random undersampling to generate a balanced data-subset
        dataset_eq = equalizedclasses(dataset,label,equalizationMethod)
        #extract a set of train and test folds from the balanced data-subset (dataset_eq)  
        cvs = constructcvs(dataset_eq,evalMethod,label,participantList,taskList)
        #create a pair containing the balanced data-subset (dataset_eq)  and the associated train and test folds (cvs)
        balancedSamples[i] = [dataset_eq,cvs]

        i = i+1

    #train and cross validate ML models based on the balanced data-subsets and the associated train and test folds
    classifierI, features_importance = classifier(dataset, features, label, balancedSamples,evalMethod)

    #extract the performance metrics
    classifier_f1.extend(classifierI['test_f1'])
    classifier_recall.extend(classifierI['test_recall'])
    classifier_accuracy.extend(classifierI['test_accuracy'])
    classifier_precision.extend(classifierI['test_precision'])

 
    # report the performance metrics
    print('--  Model')
    print('F1 score# (1) mean: {} (2) variance: {}'.format(np.mean(classifier_f1), np.var(classifier_f1)))
    print('Recall score# (1) mean: {} (2) variance: {}'.format(np.mean(classifier_recall), np.var(classifier_recall)))
    print('Accuracy score# (1) mean: {} (2) variance: {}'.format(np.mean(classifier_accuracy), np.var(classifier_accuracy)))
    print('Precision score# (1) mean: {} (2) variance: {}'.format(np.mean(classifier_precision), np.var(classifier_precision)))    


    # save feature importance data
    features_importance.to_csv(PredictionResultsfolder+"features importance/MLmetrics_"+level+"_"+evalMethod+"_"+highlights+"Highlights_detailed.csv")


    return {
    'classifier_f1':np.mean(classifier_f1),
    'classifier_recall':np.mean(classifier_recall),
    'classifier_accuracy':np.mean(classifier_accuracy),
    'classifier_precision':np.mean(classifier_precision)
    }


#load the training fold of a balanced data-subset
def downsampler(X, y,nsample,ncrossval,balancedSamples,features,label):
    X= balancedSamples[nsample][1][ncrossval][0][features]
    y= balancedSamples[nsample][1][ncrossval][0][label]
    return X,y


#get the testing folds of a sepcific cross-validation
def get_cvs(balancedSamples,numberOfSamplingIterations,ncrossval,features,label):
    cv = list()

    for nsample in range(0,numberOfSamplingIterations):
        fs =  balancedSamples[nsample][1][ncrossval][1][features]
        l =     balancedSamples[nsample][1][ncrossval][1][label]        
        cv.append({"features":fs, "label":l})

    return cv



#validate a model (by comparing the values predicted by the models with the ones expected)
def crossval(cv,model):
    if len(cv["features"])>0:
        print("crossval")
        X_test = cv["features"]

        y_pred = model.predict(X_test)
        y_true = cv["label"]

        #evaluation
        f1.append(f1_score(y_true, y_pred))
        accuracy.append(accuracy_score(y_true, y_pred))
        precision.append(precision_score(y_true, y_pred))
        recall.append(recall_score(y_true, y_pred))


#use the training sets of the balancedSamples to train a model, and the testing sets to validate that model
def crossvals(ncrossval,balancedSamples,features,label,numberOfSamplingIterations,X,y):


        print(f'*************************************************************************************** no crossval {ncrossval}')

        #create n pipelines. n: corresponds to the number of balanced data-subsets (which is equal to number of sampling iterations)
            #each pipeline
                #load the training fold of a balanced data-subset
                #select k best features in the training fold
                #train a classifier with the k best features
        pipelinetuples = [(str(nsample),make_pipeline(
        FunctionSampler(func=downsampler,kw_args={'nsample': nsample, 'ncrossval': ncrossval, 'balancedSamples':balancedSamples, 'features':features, 'label':label}), #provide sampled X, y 
        SelectKBest(score_func=mutual_info_classif, k=kbest), 
        DecisionTreeClassifier()
        )
        ) 
         for nsample in range(0,numberOfSamplingIterations)] 

        # embed the pipelines in a voting classifier model
        model = VotingClassifier(estimators=pipelinetuples)

        #fit the voting classifier model
        print("model fit")
        model.fit(X, y) #X and y will be interchanged by the downsampled ones, see FunctionSampler and downsampler implementation


        #generate and store features importance
        print("feature_importance for each estimator")
        for i in range(0, numberOfSamplingIterations):

            print(f'estimator {i}')

            estimator = model.estimators_[i]['decisiontreeclassifier']
            print(estimator.feature_importances_)

            fe = model.estimators_[i].named_steps['selectkbest']


            feature_importance = pd.DataFrame(estimator.feature_importances_,
                                                index = X.columns[fe.get_support()],
                                                columns=['importance']).sort_values('importance', ascending=False)
            
            feature_importance['feature'] = feature_importance.index
            feature_importance['round'] = f'{i};{ncrossval}'

            global features_importance
            features_importance = features_importance.append(feature_importance)


        #get the testing folds (of that sepcific cross-validation)
        print("get_csvs")

        cvs = get_cvs(balancedSamples,numberOfSamplingIterations,ncrossval,features,label)


        #validate the voting classifier model with respect to each of the testing folds (by comparing the values predicted by the models with the ones expected)
        results = Parallel(n_jobs=12, backend="threading")(delayed(crossval)(cv,model) for cv in cvs)


#train and cross validate ML models based on the balanced data-subsets and the associated train and test folds
def classifier(dataset, features, label, balancedSamples,evalMethod):


    X = dataset[features]
    y = dataset[label]

    #iterate over the possible cross-validations rounds
    for ncrossval in range(0,len(balancedSamples[0][1])):
        #for each cross-validation round, use the training sets of the balancedSamples to train a model, and the testing sets to validate that model
        crossvals(ncrossval,balancedSamples,features,label,numberOfSamplingIterations,X,y) 


    # return the performance metrics and features importance
    return {"test_f1": f1, "test_accuracy": accuracy, "test_precision": precision, "test_recall": recall}, features_importance



#allow to conduct random undersampling or random oversampling depending on the equalizationMethod
def equalizedclasses(df,col,equalizationMethod): #assuming binary classes


    equalizedclasses= None
    class_count_0, class_count_1 = df[col].value_counts()

    # Separate class
    class_0 = df[df[col] == 0]
    class_1 = df[df[col] == 1]


    if equalizationMethod=='low':
        class_0_under = class_0.sample(class_count_1)
        equalizedclasses = pd.concat([class_0_under, class_1], axis=0)

    if equalizationMethod=='high':
        class_1_over = class_1.sample(class_count_0, replace=True)
        equalizedclasses = pd.concat([class_1_over, class_0], axis=0)
    
    return equalizedclasses  

 
#extract a set of train and test folds from a given dataset
def constructcvs(dataset,evalMethod,label,participantsList,taskList):

    cv = None 

    if evalMethod=='participant':
        print("cross validation method:"+evalMethod)
        cv = list()
        for participantToTtest in participantsList:

            train =  dataset.loc[dataset['participant']!=participantToTtest]
            test =  dataset.loc[dataset['participant']==participantToTtest]
            cv.append((train, test))
     
    if evalMethod=='task':
        print("cross validation method:"+evalMethod)
        cv = list()
        for taskToTtest in taskList:

            train =  dataset.loc[dataset['task']!=taskToTtest]
            test = dataset.loc[dataset['task']==taskToTtest]
            cv.append((train, test)) 


    if evalMethod=='participant-task':
        print("cross validation method:"+evalMethod)
        cv = list()

        for participantToTtest in participantsList:
            for taskToTtest in taskList:

                train =  dataset.loc[(dataset['task']!=taskToTtest) &   (dataset['participant']!=participantToTtest)   ]
                test = dataset.loc[(dataset['task']==taskToTtest) &   (dataset['participant']==participantToTtest)     ]
                cv.append((train, test)) 
                    
    return cv 





#intiate a dataframe which will contain the results
results = pd.DataFrame(columns=['cross_validation','level', 'classifier_f1', 'classifier_recall', 'classifier_accuracy','classifier_precision'])

print("--------- params")
print(f'numberOfSamplingIterations={numberOfSamplingIterations},  prediction={prediction}, sampling={sampling}, highlights={highlights}')


if level in "Lines":

    print("------------------------------------lines Pred:")

    lineML= pd.read_csv(mldataset, dtype=special_dtypes) 

    #exclude the participant with bad data quality
    lineML = lineML[~lineML["participant"].isin(participantsToExclude)]

    #models training and validation
    resline = ml(numberOfSamplingIterations,lineML,featuresLinePred,
           'label_LineHighlighted_',
           sampling,
           prediction,'linelevel')

    #add performance metrics to the results dataframe
    results = results.append({
        'level': 'line',
        'cross_validation': prediction,
        'classifier_f1': resline['classifier_f1'],
        'classifier_recall': resline['classifier_recall'],
        'classifier_accuracy': resline['classifier_accuracy'],
        'classifier_precision': resline['classifier_precision'],        
         }, ignore_index=True)


if level in "Fragments":

    print("-----------------------------Fragment Pred:")

    fragmentML = pd.read_csv(mldataset, dtype=special_dtypes)

    #exclude the participant with bad data quality
    fragmentML = fragmentML[~fragmentML["participant"].isin(participantsToExclude)]

    #model training and validation
    restfragment = ml(numberOfSamplingIterations,fragmentML,featuresFragmentPred,
           'label_fragmentHighlighted_',
           sampling,
           prediction,'fragmentlevel')

    #add performance metrics to the results dataframe
    results = results.append({
        'level': 'fragment',
        'cross_validation': prediction,
        'classifier_f1': restfragment['classifier_f1'],
        'classifier_recall': restfragment['classifier_recall'],
        'classifier_accuracy': restfragment['classifier_accuracy'],
        'classifier_precision': restfragment['classifier_precision'],        
         }, ignore_index=True)


#export results
results.to_csv(f'{PredictionResultsfolder}results_{highlights}.csv', mode='a', header=False) if os.path.isfile(f'{PredictionResultsfolder}results_{highlights}.csv') else results.to_csv(f'{PredictionResultsfolder}results_{highlights}.csv')

