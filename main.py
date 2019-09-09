# source .venv/Scripts/activate
# .venv/Lib/site-packages/pyqt5_tools/Qt/bin/designer.exe --> start designer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_excel('data/nbfs.xlsx', header=0)
print(dataset.head(5))