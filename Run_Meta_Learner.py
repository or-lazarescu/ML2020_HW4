import os
import pandas as pd
import seaborn as sns
import xgboost as xgb
import numpy as np
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import seaborn as sns
import shap

path = os.path.join('..', 'Data', 'Meta_Features_and_Results.csv')
Meta_Data = pd.read_csv(path)
print(Meta_Data)
all_columns = Meta_Data.columns.to_list()
print(all_columns)
y_columns = ['GentleBoost', 'Gradient boosting', 'XGBoost-imbalance', 'DEBoost']#, 'ThunderSVM'
relevant_columns = all_columns
relevant_columns = [x for x in relevant_columns if x not in y_columns]
relevant_columns.remove('dataset')
print(relevant_columns)

X = Meta_Data[relevant_columns]
y = Meta_Data['XGBoost-imbalance']

print(X)
print(y)

results_dict = {'Dataset Name': [], 'Accuracy': [], 'Precision': [], 'Recall': []}

loo = LeaveOneOut()
loo.get_n_splits(X)
print(loo)
for train_index, test_index in loo.split(X):
    # print('Data set: ', Meta_Data['dataset'].iloc[test_index].values[0])
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # print(X_train, X_test, y_train, y_test)
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    # print(prediction)
    clr = classification_report(y_test, prediction, output_dict=True)
    # print(clr)
    results_dict['Dataset Name'].append(Meta_Data['dataset'].iloc[test_index].values[0])
    results_dict['Precision'].append(clr['macro avg']['precision'])
    results_dict['Accuracy'].append(clr['macro avg']['precision'])
    results_dict['Recall'].append(clr['macro avg']['recall'])

Meta_Learner_Results_XGBoostImbalance = pd.DataFrame.from_dict(results_dict)
path = os.path.join('..', 'Results', 'Meta_Learner_Results_XGBoostImbalance.csv')
Meta_Learner_Results_XGBoostImbalance.to_csv(path, index=False)
'---------------------------- Extract Importance Values --------------'
### https://towardsdatascience.com/be-careful-when-interpreting-your-features-importance-in-xgboost-6e16132588e7
model = xgb.XGBClassifier()
model.fit(X, y)
feature_importance = model.feature_importances_
# print(feature_importance)
sorted_idx = np.argsort(model.feature_importances_)[::-1]
# for index in sorted_idx:
#     print([X.columns[index], model.feature_importances_[index]])

def importance_plot(importance_dict, measure, color):

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

    features = importance_dict.keys()
    values = importance_dict.values()
    importance_df = pd.DataFrame(list(zip(features, values)),
                      columns=['Feature', 'Importance value'])

    importance_df = importance_df.sort_values(by='Importance value', ascending=False)
    print(importance_df)
    ax = sns.barplot(x="Importance value", y="Feature", data=importance_df.head(15), color=color)
    ax.tick_params(axis="y", labelsize=10)
    # ax.tick_params(axis="y", direction="in", labelsize=9, pad=-190)  # , pad=-10
    path = os.path.join('..', 'Results', 'Meta_Feature_Importance '+ measure +' plot.pdf')
    # plt.margins(1, tight=True)
    plt.title(measure,  fontdict = {'fontsize' : 20})
    plt.xlabel("Importance value", fontsize=15)
    plt.ylabel('Feature', fontsize=15)

    # plt.savefig(path, dpi=100)
    plt.close()
    # plt.show()

# xgb.plot_importance(model, max_num_features = 15)
# plt.show()
f = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
colors = ['grey', 'red', 'green', 'blue', 'orange']
c = 0
for measure in f:
    color = colors[c]
    c += 1
    importance = model.get_booster().get_score(importance_type= measure)
    print(importance)
    importance_plot(importance, measure, color)

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, show=False)
all_axes = fig.get_axes()
ax = all_axes[0]
ax.tick_params(axis="y", labelsize=4, direction='in', pad=-30)
# plt.show()
path = os.path.join('..', 'Results', 'SHAP_Meta_Feature_Importance_plot.pdf')
plt.savefig(path, dpi=100)
plt.close()

