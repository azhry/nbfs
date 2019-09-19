import numpy as np
import math

class NaiveBayes:
  
	def __init__(self):
		pass
  
	def fit(self, features, labels):
	    self.features = features
	    self.labels = labels
	    self.class_dist = self.class_probability()
	    self.meanstd = self.attribute_meanstd()
    
	def predict(self, data):
		predicted = np.empty((0, len(data)), np.int)
		for i in range(len(data)):
			attribute_proba = dict()
			class_proba = dict()
			for label in self.unique_label:
				attribute_proba[label] = dict()
				for col in self.numeric_columns:
					attribute_proba[label][col] = (np.exp(-(((data.iloc[i][col] - self.meanstd[label][col]['mean']) ** 2) / (2 * (self.meanstd[label][col]['std'] ** 2)))) / (math.sqrt(2 * math.pi) * self.meanstd[label][col]['std'])) * self.class_dist[label]
				value_proba = list(attribute_proba[label].values())
				class_proba[label] = np.prod(np.array(value_proba))
			predicted = np.append(predicted, max(class_proba.keys(), key=lambda k: class_proba[k]))

		return predicted
  
	def class_probability(self):
		self.unique_label, counts = np.unique(self.labels, return_counts=True)
		total = np.sum(counts)
		dist = [v / total for v in counts]
		dist = dict(zip(self.unique_label, dist))
		return dist
    
	def attribute_meanstd(self):
		unique, counts = np.unique(self.labels, return_counts=True)
		self.numeric_columns = self.features._get_numeric_data().columns
		dataset = self.features
		dataset['Classification'] = self.labels
    
		res = dict()
		meanstd = dict()
		for label in unique:
			res[label] = dataset[dataset['Classification'] == label]
			res[label] = res[label].drop(['Classification'], axis=1)
			meanstd[label] = dict()
			for col in self.numeric_columns:
				meanstd[label][col] = dict()
				meanstd[label][col]['mean'], meanstd[label][col]['std'] = np.mean(res[label][col]), np.std(res[label][col])

		return meanstd