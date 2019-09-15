# file:///C:/Users/Administrator/Downloads/Analisis%20Regresi%20Linear%20Berganda%20(1)-2017-10-30T11_13_45.352Z.pdf

import numpy as np, pandas as pd

class LinearRegression:

    def __init__(self, dataset):
        self.columns = dataset.columns
        self.labels = dataset.iloc[:, -1]
        self.features = dataset.drop([self.columns[-1]], axis=1)


    # def regression_table(self):
    #     columns = self.features.columns
    #     new_features = self.features.copy()
    #     for column in columns:
    #         # new_features[f'{column}_squared'] = new_features[column] ** 2
    #         # new_features[f'{column}_y'] = new_features[column] * self.labels
    #         for column_r in columns:
    #             if f'{column}_{column_r}' in new_features or f'{column_r}_{column}' in new_features:
    #                 continue
    #             new_features[f'{column}_{column_r}'] = new_features[column] * new_features[column_r]

    #     return new_features

    def regression_table(self):
        n = len(self.features)
        columns = self.features.columns
        num_columns = len(columns) + 1
        new_features = self.features.copy()
        matrix = dict()
        for i, column in enumerate(columns):
            matrix[column] = dict()
            for j, column_r in enumerate(columns):
                if f'{column}_{column_r}' in new_features or f'{column_r}_{column}' in new_features:
                    if f'{column_r}_{column}' in new_features:
                        matrix[column][column_r] = np.sum(new_features[f'{column_r}_{column}'])
                    else:
                        matrix[column][column_r] = np.sum(new_features[f'{column}_{column_r}'])
                    continue
                new_features[f'{column}_{column_r}'] = new_features[column] * new_features[column_r]
                matrix[column][column_r] = np.sum(new_features[f'{column}_{column_r}'])
        
        return matrix, new_features

        def determinant_matrix(self, matrix):
            pass