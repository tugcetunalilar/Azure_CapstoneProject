# Capstone Project

## Project Overview

The capstone project is the last part of the Azure Nanodegree Program, Machine Learning Engineer with Microsoft Azure. For this project, we were required to find an external dataset and use machine learning methods to make predictions. For this project, we used the heart failure records data that exist in Kaggle, and applied machine learning methods to it to predict the likelihood of survival of a patient who has heart failure problems. To do this, we used AutoML and another custom model that uses Logistic regression and selected the best model based on the primary metric we defined, which we defined as accuracy and compared the two models based on this. 

## Dataset

## Overview and Task

For this project, we used the data on heart failure records from Kaggle. The file contains information on 299 patients who were treated at a hospital in Punjab region of Pakistan betweeen April and December 2015. Out of the 299 patients, 105 were women and 194 were men. The age range is between 40 and 95 years old. There are additional data points on patients in the dataset that include the chronic conditions they have (diabetes, anemia, high blood pressure, smoking), levels of certain blood cells and minerals in the blood (CPK, platelets, serum creatinine, serum sodium), blood ejection fraction, follow-up days and whether the patient died during follow-up. The task was to use these various features to predict the likelihood of patient's passing during the follow-up period. 

## Access
I downloaded the csv file from [Kaggle!](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data). Then we uploaded it to the workspace, saved it as a tabular dataset and set the column names. For AutoML experiment, we read this file from the workspace. For our hyperdrive experiment, we accessed it using the Github directory and converted it to dataframe for experimental purposes.

## Automated ML:
## Settings and Configuration

Below are the settings and configuration for AutoML experiment:

- 5 concurrent iterations
- Primary metric defined as accuracy
- Task defined as classification
- Early stopping enabled as Bandit Policy


## Hyperparameter search: types of parameters and their ranges

As machine learning task at hand was of Classification, we use Sci-kit Learn Logistic Regression classifier. 
We decided to tune two hyperparameters as follows: 

i) C: This controls regularization in model i.e co-efficient values. It is inverse of regularization strength  i.e smaller values cuase stronger regularization. We tested it using Hyperdrive using a unform sample space from 0 to 1.

ii) max_iter: Maximum number of iterations allowed for model to converge. We tested it using Hyperdrive using a set sample of list values 50,100,150,200,250.

Hyperdrive parameter sampler was RamdomParameterSampling. This rendomly selects paramter values from sample space provided. 

Hyperdrive was configured to select best parameters using highest accuracy scored produced and the goal was to maximize the accuracy metric. A total of 40 model runs were alowed with 4 max concurrent runs. 


## Two models with the best parameters

i) From AutoML experiment best model selected was VotingEnsemble with 88% accuracy. Details of its parameters are as follows:
- min_child_weight=1,
- missing=nan,
- n_estimators=10,
- n_jobs=1,
- nthread=None,
- objective='reg:logistic',
- random_state=0,
- reg_alpha=0,
- reg_lambda=0.625,
- scale_pos_weight=1,
- seed=None,
- silent=None,
- subsample=1,
- tree_method='auto', 
- flatten_transform=None,
- weights=[0.125, 0.125, 0.125,  0.125, 0.125, 0.125, 0.125, 0.125]
       
         
   Run(Experiment: capstone-Automl,
Id: AutoML_8934d6c4-8831-4d4b-98e5-907b3bdab98d_40,
Type: azureml.scriptrun,
Status: Completed)

{'precision_score_weighted': 0.8958617020161247, 'average_precision_score_micro': 0.9269018734864382, 'AUC_weighted': 0.9191387579070647, 'balanced_accuracy': 0.8511186336229242, 'matthews_correlation': 0.7423301682990012, 'f1_score_weighted': 0.8784478545347623, 'precision_score_macro': 0.8941003107622102, 'precision_score_micro': 0.8831034482758622, 'f1_score_macro': 0.8596588655909961, 'recall_score_weighted': 0.8831034482758622, 'f1_score_micro': 0.8831034482758622, 'weighted_accuracy': 0.9035554026334669, 'log_loss': 0.3830698345834191, 'recall_score_macro': 0.8511186336229242, 'AUC_macro': 0.9191387579070645, 'norm_macro_recall': 0.7022372672458486, 'average_precision_score_weighted': 0.9299034435441277, 'accuracy': 0.8831034482758622, 'average_precision_score_macro': 0.9077163867341911, 'recall_score_micro': 0.8831034482758622, 'AUC_micro': 0.9237253269916765, 'confusion_matrix': 'aml://artifactId/ExperimentRun/dcid.AutoML_8934d6c4-8831-4d4b-98e5-907b3bdab98d_40/confusion_matrix', 'accuracy_table': 'aml://artifactId/ExperimentRun/dcid.AutoML_8934d6c4-8831-4d4b-98e5-907b3bdab98d_40/accuracy_table'}


ii) From Sci-kit Learn trained model , tuned with Hyperdrive, best model was  Logistic Regression with 83 % accuracy. Details of its parameters are as follows:

['--C', '0.8848572144734638', '--max_iter', '100']


## Deployed model and instructions on how to query the endpoint with a sample input

Based on higher accuracy metric produced, we selected VotingEnsemble model produced by AutoML experiment for deployment. 
In order to deploy it , we first registerd the model and provided it an environment for deployment. We took advantage of Azure provided environment "AzureML-AutoML" . 
We set up Inference Configuration and provided it with a scoring file, this file contained API model (i.e fields that API would need for data interchange).
We then deployed the model using Azure Container Instance Webservices (Aci). Deployment enabled a REST API that provide scoring uri with keys for authentication. 
We passed test data inform of Json load to webservice configured and it validated our deployment by providing a response in expected format (1,0)


## How to improve the project in the future

We can suggest following improvments that may result in better model or faster model deployment:

i) Use more powerful computer cluster such as GPU instanced with more nodes. This may enable increase in concurrent iterations.

ii) Data has class imbalance with many 1/3 deaths events vs 2/3 non deaths events. We might address it by procuring more data.

iii) We need to assess Classifiers that AutoML had not tested and hyperparameters that were not configured. We might further wish to train our data using those model and parameters by using hyperdrive run and experiment with a different set of parameters.

iv) In Hyperdrive experimnet test might use more classifiers including some of ensemble classifiers as identified by AutoML. This might improve model performance by identifying a faster and more accurate model. 

v) In Hyperdrive experiment use Bayesian Parameter sampling: This might make experiment run faster and be able to quickly identify best hyperparameter.

vii) In Hyperdrive experiment test more hyperparameter for tuning such as penalty, solver, class_weight etc. They might improve model performance by testing a hyperparameter combination that is able to yield more accurate model.

iv) In Hyperdrive experiment to address class imbalance in data by either using SMOTE resampling technique or using class_weight parameter.

vi) In Hyperdrive experiment we have now performed any feature engineering or data standarization/normalization. Conversely we have not performed any Principal Component Analysis (PCA) to identify features with best predictive powers. We might perform these steps/techniques to improve score of model on accuracy metric.

vii) We might select further types of classifiers from Sci-kit learn library like Decision Tree classifier etc and train them to get a model with better accuracy score.


## Screen shots with a short description

i) AutoML Model: 

-screenshot of the RunDetails widget that shows the progress of the training runs of the different experiments.

![](https://github.com/nabeelsana/Capstone_Project/blob/master/starter_file/AutoML_RunWidget1.png)


-screenshot of the best model based in accuracy metric with its run id.


![](https://github.com/nabeelsana/Capstone_Project/blob/master/starter_file/AutoML_BestModel2.png)


![](https://github.com/nabeelsana/Capstone_Project/blob/master/starter_file/AutoML_BestModel3.png)


ii) Hyperdrive Model:

-Screenshot of the RunDetails widget that shows the progress of the training runs of the different experiments.


![](https://github.com/nabeelsana/Capstone_Project/blob/master/starter_file/HyperDrive_Runwidget1.png)


![](https://github.com/nabeelsana/Capstone_Project/blob/master/starter_file/HyperDrive_Runwidget2.png)


![](https://github.com/nabeelsana/Capstone_Project/blob/master/starter_file/HyperDrive_Runwidget3.png)


![](https://github.com/nabeelsana/Capstone_Project/blob/master/starter_file/HyperDrive_Runwidget4.png)


-screenshot of the best model with its run id and the different hyperparameters that were tuned.


![](https://github.com/nabeelsana/Capstone_Project/blob/master/starter_file/HyperDrive_BestModel5.png)



Deploying the Model:

-screenshot showing the deployed model, endpoint as active.


![](https://github.com/nabeelsana/Capstone_Project/blob/master/starter_file/Model_EndPoint1.png)


![](https://github.com/nabeelsana/Capstone_Project/blob/master/starter_file/Model_EndPoint2.png)


-After completion of eperiment, Webservice is being deleted

![](https://github.com/nabeelsana/Capstone_Project/blob/master/starter_file/Webservicedeleted1.png)

