# Should use the `find_k_nearest_neighbors` function below.
import numpy as np


def predict_label(examples, features, k, label_key="is_intrusive"):
    K_nearest_neighbor = find_k_nearest_neighbors(examples, features, k)
    K_nearest_neighbor_label = [examples[pid][label_key] for pid in K_nearest_neighbor]
    return round(sum(K_nearest_neighbor_label) / k)


def find_k_nearest_neighbors(examples, features, k):
    distances = {}
    for pid, features_label in examples.items():
        distances[pid] = eucd_dist(features, features_label["features"])
    return sorted(distances, key=distances.get)[:k]


def eucd_dist(features1, features2):
    return np.linalg.norm(np.array(features1) - np.array(features2))


'''
Input:
{
  "method": "predict_label",
  "features": [4.30936122, 4.28739283, 4.29680938, 4.33571647, 4.28774593],
  "k": 1
}
'''
