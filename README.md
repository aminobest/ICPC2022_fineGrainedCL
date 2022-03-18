This project contains the analysis material for the ICPC paper "Estimating Developers’ Cognitive Load at a Fine-grained level Using Non-intrusive Psycho-physiological Measures"


-Folder: "processingPipeline" contains (1) the pipeline used to extract the features, (2) a sample corresponding to the data for one participant (P01) performing one task (C_5) (the full dataset (9.57GB) can be available under request), (3) a sample of the extracted metrics and a sample of the datasets used for the machine learning (ML) part

	-Structure summary:
			- Data:
				- highlights:
					- initial.csv: initial highlights used for the labeling
					- revised.csv: revised highlights used for the labeling
				- sample:
					- fixationLog.csv: a sample of fixation data corresponding to one participant (P01) performing one task (C_5)
					- pupilLog.csv: a sample of pupil data corresponding to one participant (P01) performing one task (C_5). The pupil data was cleaned following the pipeline proposed in "Stefan Zugal, Jakob Pinggera, Manuel Neurauter, Thomas Maran, and Barbara 1270 Weber. 2017. Cheetah experimental platform web 1.0: cleaning pupillary data. 1271 arXiv preprint arXiv:1703.09468 (2017)."
				- lib (this folder contains the scripts used in the features extraction pipeline):
					- main.py: entry point to the  features extraction pipeline (all the other python files within lib are called from there)
				- out:
					- fixationLog_metrics.csv, pupilLog_metrics.csv: samples of the extracted metrics
					- sampleMLDataFragments.csv,sampleMLDataLines.csv: samples of the datasets used to train and test ML models to estimate the mentally demanding fragments and lines of code respectively



-Folder: "ML" contains the script used to train the machine learning (ML) models, the features/labels datasets and the cross-validation results

	-Structure summary:
		- datasets:
			- initial:
				- MLDataFragments: ML dataset with features computed at the fragment level and labels obtained from the initial highlights
				- MLDataLines: ML dataset with features computed at the line level and labels obtained from the initial highlights
			- revised:
				- MLDataFragments: ML dataset with features computed at the fragment level and labels obtained from the revised highlights
				- MLDataLines: ML dataset with features computed at the line level and labels obtained from the revised highlights
		- lib:
			- ML.py: script to use to train and cross validate the ML models
		- out:
			- initial:
				- results_initial.csv: performance measures (F1, recall, accuracy and precision) computed for models trained and tested with labels obtained from the initial highlights
				- feature importance: feature importance scores for models trained and tested with labels obtained from the initial highlights
			- revised: 
				- results_revised.csv: performance measures (F1, recall, accuracy and precision) computed for models trained and tested with labels obtained from the revised highlights
				- feature importance: feature importance scores for models trained and tested with labels obtained from the revised highlights

-Folder: "data collection material" contains the material used for the data collection including (1) the used comprehension tasks (2) the highlights tasks and (3) the quiz
	
	- Structure summary:
		- code comprehension tasks: the Java source-code files used as code comprehension tasks
		- highlighting tasks: the code comprehension tasks shown in word files, allowing the participants to highlight the mentally demanding parts of code
		- quiz: the tasks used as a quiz to ensure that the participants have the necessary background to take the experiment


- Environnement
	- Python 3.9.5 
	- External python modules required for code execution
		- numpy, pandas, sklearn, imblearn, os, sys, random, joblib, networkx, pywt, math, scipy, re




