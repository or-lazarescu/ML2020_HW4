import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
import numpy as np
import scikit_posthocs as sp


# path = os.path.join('..', 'Results', 'updated results.csv')
# Results_1 = pd.read_csv(path)
# print('Results_1', Results_1.shape)
#
# path = os.path.join('..', 'Results', 'Thunder_Datasets_Results.csv')
# Results_2 = pd.read_csv(path)
# print('Results_2', Results_2.shape)
#
# path = os.path.join('..', 'Results', 'gpboost_Results.csv')
# Results_3 = pd.read_csv(path)
# print('Results_3', Results_3.shape)
#
# Results = pd.concat([Results_1, Results_2, Results_3], ignore_index=True, axis=1)
# print('Results', Results.shape)

path = os.path.join('..', 'Results', 'Classification results (5 algorithms 150 datasets).csv')
Results = pd.read_csv(path)
print('Results', Results.shape)

# print(list(Results))
# path = os.path.join('..', 'Results', 'updated results.csv')
# Results = pd.read_csv(path)
print(Results)
print(list(Results))

measurements = ['Accuracy', 'TPR', 'FPR', 'Precision', 'AUC', 'PR - Curve', 'Training Time', 'Inference Time']


# for m in measurements:
#     sns.boxplot(x='Algorithm Name', y=m, data=Results, showfliers=False)
#     plt.show()



algorithms_names = Results['Algorithm Name'].unique()
print(algorithms_names)
measures_list = []
for algorithm in algorithms_names:
    measures = Results['AUC'][Results['Algorithm Name'] == algorithm].to_list()
    measures_list.append(measures)
    print(len(measures))

print(measures_list)
measures_array = np.array(measures_list)
print(len(measures_array))
fridman_results = friedmanchisquare(*measures_array)
print(fridman_results)


'------------------------- Post Hoc test ----------------------------'


pc = sp.posthoc_nemenyi(Results, val_col='AUC', group_col='Algorithm Name')#, p_adjust = 'holm'
cmap = ['1', '#fb6a4a',  '#08306b',  '#4292c6', '#c6dbef']
heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
sp.sign_plot(pc, **heatmap_args)
plt.xticks(rotation=45)
plt.show()

"""
pc = sp.posthoc_nemenyi_friedman(measures_list)
print(pc)
heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
sp.sign_plot(pc, **heatmap_args)


ranks = np.array([rankdata(-p) for p in measures_list])
average_ranks = np.mean(ranks, axis=0)
print('\n'.join('{} average rank: {}'.format(a, r) for a, r in zip(algorithms_names, average_ranks)))

cd = compute_CD(average_ranks,
n=len(measures_list),
alpha='0.1',
test='nemenyi')
# This method generates the plot.
graph_ranks(average_ranks,
names=algorithms_names,
cd=cd,
width=10,
textspace=1.5,
reverse=True)
plt.show()
"""
