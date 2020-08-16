"""
Mission:

A. Load and import all relevant algorithms + GBM for comparision.

B. Parameter tuning.

C. 10-fold cross validation on 150 data sets.

D. Fridman test to find best algorithm.

E. Meta learner based on running results and meta features data.

"""




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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


# data sets directory
path = os.path.join('..', 'Data', 'classification_datasets')
data_files_list = os.listdir(path)
print('data_files_list:', data_files_list)

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


    y = One_dataset.iloc[:,-1]
    print(y)
    X = One_dataset.iloc[:, :-1]
    print(X)

    # Split to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

    # Run the model
    rgr = DEBoostClassifier(method='classification')
    rgr.fit(X_train, y_train)
    predictions = rgr.predict(X_test)

    print(predictions)
    clr = classification_report(y_test, predictions, output_dict=True)
    clr_df = pd.DataFrame.from_dict(clr)
    # break
    print('clr_df', clr_df)
    # if c > 4:
    #     break

def starboost_predictor(X, y):
    model2 = sb.BoostingRegressor(
        base_estimator=tree.DecisionTreeRegressor(max_depth=3),
        n_estimators=30,
        learning_rate=0.1
    )
    model2 = model2.fit(X, y)
    y_pred = model2.predict(X)
    print(y_pred)