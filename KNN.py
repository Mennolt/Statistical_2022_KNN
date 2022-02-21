import numpy as np
from statistics import mode
import time


class KNN:
    def __init__(self, train_data, k : int):
        """
        Initializes KNN class

        Inputs:
        train_data: the training data, of which column 1 is the labels
        k: the number of nearest neighbours to consider
        """

        self.train_data = train_data
        self.k = k

    def predict_point(self, point) -> int:
        """
        Predicts the class of a single datapoint.
        Input:
        point: a single datapoint of the same dimensions as the training data-labels

        Output: A label that occurs most often in the nearest neighbours
        """
        start = time.time()
        distances = self.calc_dists(point)
        print(f"distances {time.time()-start}")
        #print(distances)
        shortest_distances = self.get_lowest(distances)
        #print(sort_distances)
        print(f"shortest distances {time.time() - start}")
        #pick label that occurs most
        return mode(shortest_distances)

    def calc_dists(self, point) -> np.ndarray:
        """
        Calculates euclidean distances between input point and all other points
        :param point: a single datapoint of same dimensions as training data-labels
        :return: array of distances of each point in training data to new point,
                 with associated labels of training points
        """
        distances = np.linalg.norm(self.train_data[:, 1:] - point, ord=2, axis=1)
        #print(distances)

        return(np.c_[distances, self.train_data[:,0]])

    def get_lowest(self, distance_list : np.ndarray):
        """
        Takes an array of distances and finds the lowest k of them
        :param distance_list: ndarray containing pairs of distance, label
        :return: a list containing the k labels for the k lowest distances
        """
        lowest_dist = [distance_list[0, 0]]
        lowest_labels = [distance_list[0, 1]]

        for val in distance_list[1:]:
            i = 0
            inserted = False
            while i < self.k and i < len(lowest_dist) and inserted == False:
                if val[0] < lowest_dist[i]:
                    lowest_dist.insert(i, val[0])
                    lowest_labels.insert(i, val[1])
                    inserted = True
                i += 1
            if inserted == False and i < self.k:
                lowest_dist.append(val[0])
                lowest_labels.append(val[1])
        return lowest_labels[:self.k]


