"""
metrics module
"""

import numpy as np

from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=1)


def nn_score(trainX, testX):
    """
    Nearest neighbour eror is meassured between real and 
    generated samples
    
    :type trainX: np.array
    :param trainX: Array of real images
    
    :type testX: np.array
    :param testX: Array of generates images
    
    """
    assert len(trainX.shape) == 2 and len(testX.shape) == 2, 'Only \
            2 dimensional arryes are permitted'
    
    nn.fit(trainX)
    dist = nn.kneighbors(testX)
    
    return np.mean(dist[0])