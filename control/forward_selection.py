import math, numpy as np

class ForwardSelection:

    def __init__(self, dataset):
        self.dataset = dataset

    def correlation(self, x, y):
        n = len(x)
        return ((n * np.dot(x, y)) - (np.sum(x) * np.sum(y))) / (math.sqrt((n * np.sum(x ** 2)) - (np.sum(x) ** 2)) * math.sqrt((n * np.sum(y ** 2)) - (np.sum(y) ** 2)))

    def pearson_correlation(self, x1, x2):
        n = len(x1)
        x1_mean = np.mean(x1)
        x2_mean = np.mean(x2)
        return np.sum((x1 - x1_mean) * (x2 - x2_mean)) / math.sqrt(np.sum((x1 - x1_mean) ** 2) * np.sum((x2 - x2_mean) ** 2))