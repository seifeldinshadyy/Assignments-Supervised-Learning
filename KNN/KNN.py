# Description: K-Nearest Neighbors algorithm implementation     
import numpy as np
from collections import Counter
from euclidean_distance import euclidean_distance
from manhattan_distance import manhattan_distance
from cosine_similarity import cosine_similarity

# KNN Class
class KNN:
    def __init__(self, k=3, distance_metric="euclidean"):
        self.k = k
        self.distance_metric = distance_metric
        self.distance_function = self.get_distance_function()

    def get_distance_function(self):
        if self.distance_metric == "euclidean":
            return euclidean_distance
        elif self.distance_metric == "cosine":
            return lambda x1, x2: 1 - cosine_similarity(x1, x2)  # Convert similarity to distance
        elif self.distance_metric == "manhattan":
            return manhattan_distance
        else:
            raise ValueError("Invalid distance metric. Choose from: euclidean, cosine, manhattan.")

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        distances = [self.distance_function(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]