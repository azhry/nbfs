import math, numpy as np

class ForwardSelection:

    def __init__(self, dataset):
        self.dataset = dataset

    def correlation(self, x, y):
        n = len(x)
        return ((n * np.dot(x, y)) - (np.sum(x) * np.sum(y))) / (math.sqrt((n * np.sum(x ** 2)) - (np.sum(x) ** 2)) * math.sqrt((n * np.sum(y ** 2)) - (np.sum(y) ** 2)))

    def f_test(self):
        pass

    def ssr(self):
        pass

    def sse(self):
        pass