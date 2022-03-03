import numpy as np
from scipy.spatial import distance_matrix

def Minkowski_distance(point, train_array, p):
    '''
    Returns an array of Minkowski distances between given point p and the points in p_array. Distances are in the same
    order as the given p_array
    :param point: 1d np array, the length of which is equal to the number of features f of the point
    :param train_array: 2d np array, of size nr of datapoint x f
    :param p: int. Denotes the order of the Minkowski distance
    
    :returns: 1d np array, the length of which is equal to the nr of datapoints.
    '''
    point = point.reshape((1,-1)) #turn 1d array into a 2d array because scipy's distance_matrix only takes matrices as input
    distances = distance_matrix(point, train_array, p=p)
    
    return distances.ravel() #return the distances in a 1d array
    
    