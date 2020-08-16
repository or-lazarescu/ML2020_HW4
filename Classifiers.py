# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 13:00:47 2020

@author: juman
"""



#%%
'''
    Classifiers import packages:
        1. DEBoost
        2. GentleBoost (skboost)
        3. Gradient Boosting
        4. thundersvm
        5. imbalance_xgboost
        6. gpboost
'''

from deboost import DEBoostClassifier
from skboostmaster.skboost.gentleboost import GentleBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from thundersvm import SVC
from imxgboost.imbalance_xgb import imbalance_xgboost as imb_xgb
import gpboost as gpb


#%%
    
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import statistics

#%%
'''
    Importing useful functions from sklearn package
'''
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
#%%
"""
    HELP FUNCTIONS
"""

def preprocessing_dataset(dataset):
    numeric_columns = dataset.select_dtypes('number').columns
    dataset[numeric_columns] = dataset[numeric_columns].astype(float)
    non_numeric_columns = dataset.select_dtypes(exclude=['int', 'float']).columns
    print('non_numeric_columns: ', non_numeric_columns)
    print('numeric_columns: ', numeric_columns)
    dataset[non_numeric_columns] = dataset[non_numeric_columns].apply(lambda col: pd.Categorical(col).codes).replace(-1, np.nan)
    print(dataset)
    dataset[dataset.columns[-1]] = dataset[dataset.columns[-1]].astype(int)

    dataset = dataset.fillna(dataset.mean())
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:,-1]
    
    return X,y

def finding_hyperparameter(model,X,y):
 

    folds = 3
    param_comb = 50

    skf = StratifiedShuffleSplit(n_splits=folds, random_state = 1001)
    
    if model['label']== 'XGBoost-imbalance':
        X=X.to_numpy()
        y=np.array(y)

    random_search = RandomizedSearchCV(model['model'], param_distributions= model['params'], n_iter=param_comb,
                                        n_jobs=1, cv=skf.split(X,y), verbose=0, random_state=1000 )

   
    random_search.fit(X,y)
    # print('\n All results:')
    # print(random_search.cv_results_)
    # print('\n Best estimator:')
    # print(random_search.best_estimator_)
    # print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
    # print(random_search.best_score_ * 2 - 1)
    print('\n Best hyperparameters:')
    print(random_search.best_params_)
    return random_search.best_params_



def ten_fold_cross_validation(m,X,y_array,best_params):
    '''
    Cross validation: 10 fold. 
    calculation of accuracy, TPR, FPR, Precision, AUC-ROC, precision-recall curve, training time and inference time.
    '''
    
    results={'Fold number': [], 'Accuracy':[],'FPR':[],'TPR':[],'AUC-ROC':[],'Precision':[],'PR-curve':[],'Training Time':[],'Inference Time':[]}
    
    kf = StratifiedShuffleSplit(n_splits=10) 
    model = m['model']
    fold = 1
    for train_index, test_index in kf.split(X, y_array[0]):
        
        print('-----------> fold number '+str(fold)+'<-----------')
        

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        
        accuracy=[]
        fpr_average=[]
        tpr_average=[]
        roc_auc=[]
        precision_average=[]
        pr_curve=[]
        t_time=[]
        i_time=[]

        for i in range(len(y_array)):
            
            y = np.array(y_array[i])
            y_train, y_test = y[train_index], y[test_index]

            if len(np.unique(y_test)) ==1:
                y_test[0] = (y_test[0]-1)*-1
            
            # Fitting model and predicting
            t0 = time.time()
            if m['label'] == 'XGBoost-imbalance':
                model.fit(X_train.to_numpy(), y_train) 
            else:
                model.fit(X_train, y_train)  # train the model
            t1 = time.time()
            if m['label']=='GentleBoost':
                y_pred = model.predict_proba(X_test)
            elif m['label'] in ['DEBoost','ThunderSVM','GPBoost']:
                y_pred = model.predict(X_test)
            elif m['label']=='LiquidSVM':
                y_pred = model.test(X_test)
            elif m['label'] == 'XGBoost-imbalance':
                y_pred = model.predict_sigmoid(X_test.to_numpy())
                print(y_pred)
            else:
                y_pred = model.predict_proba(X_test)[:,1]
            t2 = time.time()
         
            
            # Calculating parameters

            accuracy.append(accuracy_score(y_test, y_pred.round()))
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            fpr_average.append(fpr[int(len(fpr)/2)])
            tpr_average.append(tpr[int(len(fpr)/2)])
            roc_auc.append(roc_auc_score(y_test, y_pred))
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
            precision_average.append(precision[int(len(precision)/2)])
            pr_curve.append(auc(recall, precision))
            t_time.append(t1-t0)
            i_time.append(((t2-t1)/len(test_index))*1000)
            
        results['Fold number'].append(fold)
        results['Accuracy'].append(statistics.mean(accuracy))
        results['FPR'].append(statistics.mean(fpr_average))
        results['TPR'].append(statistics.mean(tpr_average))
        results['AUC-ROC'].append(statistics.mean(roc_auc))
        results['Precision'].append(statistics.mean(precision_average))
        results['PR-curve'].append(statistics.mean(pr_curve))
        results['FPR'].append(statistics.mean(t_time))
        results['Training Time'].append(statistics.mean(i_time))
        results['Inference Time'].append(statistics.mean(fpr_average))
        fold+=1
    return results
        
        

def multi_class_problem(y_array,unique_class_values):
    y_binary_class = []
    if len(unique_class_values)>2:             #Multi problem

        if len(unique_class_values)>8:      #More than 7 classes

            y_fixed=[]
            average_value=statistics.mean(y_array)
    
            for i in y_array:
                if i < average_value:
                    y_fixed.append(0)
                else:
                    y_fixed.append(1)
            y_binary_class.append(y_fixed)
            
        else:

            for uValue in unique_class_values:
                y_per_class = []
                for val in y_array:
                    if val == uValue:
                        y_per_class.append(1)
                    else:
                        y_per_class.append(0)
                y_binary_class.append(y_per_class)
                
    else:                                   # Binary problem                           
        y_binary_class.append(y.to_list())   

    return y_binary_class




#%%

path = os.path.join('classification_datasets')
data_files_list = os.listdir(path)
data_files_list = sorted(data_files_list)

#%%







models = [
                {'label':'GentleBoost','model':GentleBoostClassifier(),'params':{
                    'n_estimators' : [100,150,200,500,1000],
                    'learning_rate':[0.001,0.1,0.5,1.],
                    'random_state':[None,100,300,500,1000]}},
                {'label':'Gradient boosting','model':GradientBoostingClassifier(),'params':{
                    'learning_rate':[0.05,0.1,0.2,0.3],
                          'n_estimators':[100,200,300,500],
                          'subsample':[0.1,0.3,0.5,0.8,1.0],
                          'min_samples_split':[2,5,10],
                          'min_samples_leaf':[1,3,5,10]}},
                {'label':'DEBoost','model':DEBoostClassifier(),'params':{}},
                {'label':'ThunderSVM', 'model':SVC(),'params':{
                    'kernel':['linear','rbf'],
                          'gamma':[1,3,5],
                          'C':[1.0,1.5,2.0,2.5,3.0],
                          'shrinking':[True,False]
                    }},
                {'label':'XGBoost-imbalance','model':imb_xgb(), 'params':{
                    'max_depth':[3,5,10,15,20],
                            'eta':[0.1,0.3,0.5],
                            'focal_gamma':[1.0,1.5,2.0,2.5,3.0],
                            'imbalance_alpha':[1.5,2.0,2.5,3.0,4.0]
                          
                    }},
                {'label':'GPBoost','model': gpb.GPBoostClassifier(), 'params': {
                            'learning_rate': [0.01, 0.1, 1],
                            'n_estimators': [20, 50],
                            'max_depth': [1, 5, 10]
                }}
              
        ]
results_dict = {'Dataset Name': [], 'Algorithm Name': [], 'Cross Validation': [],'Hyper-Parameters Values':[],  'Accuracy': [], 
                'TPR': [], 'FPR': [], 'Precision': [], 'AUC': [], 'PR - Curve': [], 'Training Time': [], 'Inference Time': []}


c=0
for file_name in data_files_list:
    c += 1
    print('----------- Dataset: ', file_name.split('.')[0], ' Number', c, '---------------')

    # Load data set.
    dataset_path = os.path.join('classification_datasets', file_name)
    dataset = pd.read_csv(dataset_path, engine='python')
    X,y = preprocessing_dataset(dataset)
    y_array = y.to_numpy()
    unique_class_values = np.unique(y_array)
    
    y_binary_class=[]
    
    y_binary_class = multi_class_problem(y_array,unique_class_values)
    
   
        
    for m in models:
        print(m['label'])
        best_params=''
        if m['label']!='DEBoost':
            best_params = finding_hyperparameter(m, X, y_binary_class[0])
      
        results = ten_fold_cross_validation(m, X, y_binary_class,best_params)

    
        for i in range(0,10):
            results_dict['Dataset Name'].append(file_name.split('.')[0])
            results_dict['Algorithm Name'].append(m['label'])
            results_dict['Cross Validation'].append(results['Fold number'][i])
            results_dict['Hyper-Parameters Values'].append(best_params)
            results_dict['Accuracy'].append(results['Accuracy'][i])
            results_dict['TPR'].append(results['TPR'][i])
            results_dict['FPR'].append(results['FPR'][i])
            results_dict['Precision'].append(results['Precision'][i])
            results_dict['AUC'].append(results['AUC-ROC'][i])
            results_dict['PR - Curve'].append(results['PR-curve'][i])
            results_dict['Training Time'].append(results['Training Time'][i])
            results_dict['Inference Time'].append(results['Inference Time'][i])
    
    Results_df = pd.DataFrame.from_dict(results_dict)
    Results_df.to_csv('Toy_All_Datasets_Results.csv', index=False)  








