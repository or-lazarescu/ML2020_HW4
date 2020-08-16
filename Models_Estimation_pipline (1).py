"""
Mission:

A. Load and import all relevant algorithms + GBM for comparision.

B. Parameter tuning.

C. 10-fold cross validation on 150 data sets.

D. Fridman test to find best algorithm.

E. Meta learner based on running results and meta features data.

"""



import time
import os
import pandas as pd
import lightgbm as lgbm
# import gpboost as gpb
import deboost
import starboost
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from deboost import DEBoostClassifier
from sklearn import tree
import starboost as sb
# from thundersvm import SVC
from liquidSVM import *
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
import re
from io import StringIO

# data sets directory
path = os.path.join('..', 'Data', 'classification_datasets')
data_files_list = os.listdir(path)
print('data_files_list:', data_files_list)

def finding_hyperparameter(X_data, Y_data):
    models = [ {'label': 'Random Forest', 'model': RandomForestClassifier(random_state=42),
               'params': {
                   'bootstrap': [True, False],
                   'max_depth': [10, 20, 30, 40, 50, None],
                   'min_samples_leaf': [1, 2, 3, 4, 5, 10],
                   'min_samples_split': [2, 5, 10],
                   'n_estimators': [100, 150, 200, 300, 400, 600]}},

              {'label': 'Gradient Boosting', 'model': GradientBoostingClassifier(),
               'params': {
                   'learning_rate': [0.05, 0.1, 0.2, 0.3],
                   'n_estimators': [100, 200, 300, 500],
                   'subsample': [0.1, 0.3, 0.5, 0.8, 1.0],
                   'min_samples_split': [2, 5, 10],
                   'min_samples_leaf': [1, 3, 5, 10]}}]

    best_parameters_dict = {}

    for m in models:
        print('------------', m['label'], '---------------')

        folds = 3
        param_comb = 15

        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)

        random_search = RandomizedSearchCV(m['model'], param_distributions=m['params'], n_iter=param_comb,
                                           scoring='roc_auc', n_jobs=4, cv=skf.split(X_data, Y_data), verbose=3,
                                           random_state=1001)

        # Here we go

        random_search.fit(X_data, Y_data)
        print('\n All results:')
        print(random_search.cv_results_)
        print('\n Best estimator:')
        print(random_search.best_estimator_)
        print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
        print(random_search.best_score_ * 2 - 1)
        print('\n Best hyperparameters:')
        print(random_search.best_params_)
        results = pd.DataFrame(random_search.cv_results_)
        results.to_csv(str(m['label']) + '-random-search-results-01.csv', index=False)
        best_parameters_dict[m['model']] = random_search.best_params_

    return best_parameters_dict

def is_multi_class(Label):
    num_classes = Label.nunique()
    problem_type = 'Binary'
    if num_classes > 2:
        problem_type = 'Multi Class'
        Label = pd.get_dummies(Label)
    else:
        Label = Label.to_frame()
    # print(problem_type)
    return problem_type, Label

def findMiddle(list):
  list = list.tolist()
  l = len(list)
  # print('len', l)
  if l%2 != 0:
    return (list[l//2-1]+list[l//2])/2.0
  else:
    return list[l//2 - 1]

def calculate_inference_time(trained_model, X_test, y_test):
    # print(len(y_test))
    if len(y_test) >= 1000:
        synthetic_y = y_test[:999]
        synthetic_x = X_test[:999]

    else:
        synthetic_x = pd.concat([X_test.head(10)] * 100, ignore_index=True)
        synthetic_df = pd.DataFrame({'class': np.repeat(y_test[:10], 100)})
        synthetic_y = synthetic_df['class']
    print('synthetic_x', synthetic_x)
    print('synthetic_y', synthetic_y)
    start_testing_Time = time.time()
    trained_model.predict(synthetic_x)
    testing_Time = time.time() - start_testing_Time

    return testing_Time

results_dict = {'Dataset Name': [], 'Algorithm Name': [], 'Class_Number':[], 'Cross Validation': [], 'Hyper - Parameters Values': [], 'Accuracy': [], 'TPR': [], 'FPR': [], 'Precision': [], 'AUC': [], 'PR - Curve': [], 'Training Time': [], 'Inference Time': []}
algorithms = ['RF', 'GBM']
c = 0
for file_name in data_files_list:

    # file_name = 'analcatdata_asbestos.csv'
    c += 1
    print('----------- Dataset: ', file_name.split('.')[0], ' Number', c, '---------------')
    # Load data set.
    one_path = os.path.join('..', 'Data', 'classification_datasets', file_name)
    One_dataset = pd.read_csv(one_path, engine='python')
    print(One_dataset)

    # Pre-processing data

    numeric_columns = One_dataset.select_dtypes('number').columns
    One_dataset[numeric_columns] = One_dataset[numeric_columns].astype(float)
    non_numeric_columns = One_dataset.select_dtypes(exclude=['int', 'float']).columns
    print('non_numeric_columns: ', non_numeric_columns)
    print('numeric_columns: ', numeric_columns)
    One_dataset[non_numeric_columns] = One_dataset[non_numeric_columns].apply(lambda col: pd.Categorical(col).codes).replace(-1, np.nan)
    One_dataset[One_dataset.columns[-1]] = One_dataset[One_dataset.columns[-1]].astype(int)

    print(One_dataset)
    One_dataset = One_dataset.fillna(One_dataset.mean())
    print(One_dataset)


    Label = One_dataset.iloc[:,-1]
    print(Label)
    X = One_dataset.iloc[:, :-1]
    print(X)
    problem_type, Label = is_multi_class(Label)
    print(problem_type)
    print('Label', type(Label))
    class_number = -1
    for col in Label.columns.to_list():
        class_number += 1
        print('col', col)
        if len(Label[Label[col] == 1]) > 10:
            y = Label[col]

            # Split to train and test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

            best_parameters_dict = finding_hyperparameter(X_train, y_train)
            print(best_parameters_dict)

            for model in best_parameters_dict:
                model.set_params(**best_parameters_dict[model])

                kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # n_splits=10
                tprs = []
                fprs = []
                aucs = []

                fold = 0
                for train, test in kf.split(X, y):
                    fold += 1
                    print('------------------', 'Dataset: ', file_name.split('.')[0], 'fold ', fold, '--------------------')
                    train1 = train.tolist()
                    test1 = test.tolist()
                    X_train, X_test = X.iloc[train1], X.iloc[test1]
                    y_train, y_test = y.iloc[train1], y.iloc[test1]
                    start_training_Time = time.time()
                    model.fit(X_train, y_train)  # train the model
                    training_Time = time.time() - start_training_Time
                    testing_Time = calculate_inference_time(model, X_test, y_test)

                    y_pred = model.predict(X_test)  # predict the test data
                    predictions_proba = model.predict_proba(X_test)
                    clr = classification_report(y_test, y_pred, output_dict=True)
                    print('clr', clr['macro avg'])

                    print(model.classes_)
                    # plot_ROC(y_test, predictions_proba[:, 1], model)
                    try:
                        roc_auc1 = roc_auc_score(y_test, predictions_proba[:, 1])
                        print('AUC = %0.2f' % roc_auc1)
                    except:
                        roc_auc1 = None
                    # results_dict = {'precision': clr['macro avg']['precision'], 'recall': clr['macro avg']['recall'], 'f1_score': clr['macro avg']['f1-score'], 'AUC': roc_auc1}

                    print('\n')
                    fpr, tpr, _ = roc_curve(y_test, predictions_proba[:, 1], pos_label=model.classes_[1])
                    print('tpr', tpr)
                    print('fpr', fpr)
                    middel_tpr = findMiddle(tpr)
                    middel_fpr = findMiddle(fpr)
                    print('middel_tpr', middel_tpr)
                    print('middel_fpr', middel_fpr)
                    prec, recall, _ = precision_recall_curve(y_test, predictions_proba[:, 1], pos_label=model.classes_[1])
                    accuracy = accuracy_score(y_test, y_pred)
                    average_precision = average_precision_score(y_test, predictions_proba[:, 1])
                    print(accuracy)
                    print(average_precision)
                    "------------------------ Update Results dict ------------------------------"

                    results_dict['Dataset Name'].append(file_name.split('.')[0])
                    results_dict['Algorithm Name'].append(model.__class__.__name__)
                    results_dict['Class_Number'].append(class_number)
                    results_dict['Cross Validation'].append(fold)
                    results_dict['Hyper - Parameters Values'].append(best_parameters_dict[model])
                    results_dict['AUC'].append(roc_auc1)
                    results_dict['Precision'].append(clr['macro avg']['precision'])
                    results_dict['Accuracy'].append(accuracy)
                    results_dict['TPR'].append(middel_tpr)
                    results_dict['FPR'].append(middel_fpr)
                    results_dict['PR - Curve'].append(average_precision)
                    results_dict['Training Time'].append(training_Time)
                    results_dict['Inference Time'].append(testing_Time)
    #             break
    #         break
    #     break
    # break

Results_df = pd.DataFrame.from_dict(results_dict)
print(Results_df)
Results_df.to_csv('Toy3_All_Datasets_Results.csv', index=False)
