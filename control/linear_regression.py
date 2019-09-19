# file:///C:/Users/Administrator/Downloads/Analisis%20Regresi%20Linear%20Berganda%20(1)-2017-10-30T11_13_45.352Z.pdf

import numpy as np, pandas as pd

# x[:,1] = [11, 12, 13] --> change column values

class LinearRegression:

    def __init__(self, dataset):
        self.columns = dataset.columns
        self.labels = dataset.iloc[:, -1]
        self.features = dataset.drop([self.columns[-1]], axis=1)


    def regression_table(self):
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

    def determinant_matrix(self):
        columns = self.features.columns
        matrix_size = len(columns) + 1
        new_matrix = np.empty((matrix_size, matrix_size), np.double)
        y_matrix = np.array([])

        new_matrix[0][0] = len(self.features)
        y_matrix = np.append(y_matrix, np.sum(self.labels))

        for i, column in enumerate(columns):
            new_matrix[0][i + 1] = np.sum(self.features[column])
            new_matrix[i + 1][0] = np.sum(self.features[column])
            y_matrix = np.append(y_matrix, np.sum(self.features[column] * self.labels))

            for j, column_r in enumerate(columns):
                new_matrix[i + 1][j + 1] = np.sum(self.features[column] * self.features[column_r])

        return new_matrix, y_matrix

    def determinant(self):
        x_matrix, y_matrix = self.determinant_matrix()
        matrix_size = len(x_matrix)
        b = np.array([])
        b = np.append(b, np.linalg.det(x_matrix))
        for i in range(matrix_size):
            matrix = x_matrix.copy()
            matrix[:, i] = y_matrix
            b = np.append(b, np.linalg.det(matrix))
        return b
        
