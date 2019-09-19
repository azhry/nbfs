# https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/

# source .venv/Scripts/activate
# .venv/Lib/site-packages/pyqt5_tools/Qt/bin/designer.exe --> start designer

import pandas as pd
import numpy as np
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
print(y_matrix)
print(lr.determinant())

ds = dataset.copy()
labels = ds['Classification']
features = ds.drop(['Classification'], axis=1)
nb = NaiveBayes()
nb.fit(features, labels)
predicted = nb.predict(features)
ds['Predicted'] = predicted

pd.set_option('display.max_rows', None)
print(ds)