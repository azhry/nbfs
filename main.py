# https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/

# source .venv/Scripts/activate
# .venv/Lib/site-packages/pyqt5_tools/Qt/bin/designer.exe --> start designer

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from control.forward_selection import ForwardSelection
from control.linear_regression import LinearRegression
from control.naive_bayes import NaiveBayes

pd.set_option('display.max_columns', None)

dataset = pd.read_excel('data/nbfs.xlsx', header=0)
print(dataset.head(5))

fs = ForwardSelection(dataset)
lr = LinearRegression(dataset)
det_matrix, y_matrix = lr.determinant_matrix()
lr.determinant()

determinants = lr.determinant()
regressions = np.array([])
ds = dataset.copy()
labels = ds['Classification']
features = ds.drop(['Classification'], axis=1)
shapes = features.shape
for feature in features.to_numpy():
	regressions = np.append(regressions, lr.regression(feature))

nb = NaiveBayes()
nb.fit(features, labels)
predicted = nb.predict(features)
ds['Predicted'] = predicted
ds['Y^'] = regressions
ds['Y'] = ds['Classification']
ds['(Yi-Y^)'] = (ds['Y'] - ds['Y^'])
ds['(Yi-Y^)2'] = (ds['Y'] - ds['Y^']) ** 2
ds['(Y^-Ym)'] = (ds['Y^'] - np.mean(ds['Y']))
ds['(Y^-Ym)2'] = ds['(Y^-Ym)'] ** 2

pd.set_option('display.max_rows', None)
print(ds)

sse = np.sum(ds['(Yi-Y^)2'])
ssr = np.sum(ds['(Y^-Ym)2'])
ssse = math.sqrt(sse / (len(ds) - (shapes[1] + 1)))
pearson = fs.pearson_correlation(ds['Age'], ds['BMI'])

t = ssse / math.sqrt((np.sum(ds['Age'] ** 2) - (((np.sum(ds['Age']) / shapes[0]) ** 2) * shapes[0])) * (1 - (pearson ** 2)))
f_hitung = determinants[1] / t
print(sse, ssr, ssse, pearson, t, f_hitung)