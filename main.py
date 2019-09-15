# source .venv/Scripts/activate
# .venv/Lib/site-packages/pyqt5_tools/Qt/bin/designer.exe --> start designer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from control.forward_selection import ForwardSelection
from control.linear_regression import LinearRegression

pd.set_option('display.max_columns', None)

dataset = pd.read_excel('data/nbfs.xlsx', header=0)
print(dataset.head(5))

fs = ForwardSelection(dataset)
lr = LinearRegression(dataset)
print(lr.regression_table()[0])

# columns = np.array([])
# filtered_dataset = dataset.copy()
# results = dict()
# for col in filtered_dataset.columns:
#     if col != 'Classification':
#         columns = np.append(columns, col)
#         results[col] = {
#             "correlation": fs.correlation(filtered_dataset[col], filtered_dataset['Classification']),
#             "correlation_squared": fs.correlation(filtered_dataset[col], filtered_dataset['Classification']) ** 2,
#             "std": np.std(filtered_dataset[col])
#         }

# print(results)