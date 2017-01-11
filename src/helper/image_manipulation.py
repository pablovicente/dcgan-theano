"""
image_manipulation module
"""

import theano
import numpy as np

rng = np.random.RandomState(42)

def transform(X):
    """
    Inputs images are scale to [-1,1] according to 
    the paper. 
    
    :type X: np.array
    :param X: Array of images
    """

    return np.asarray(X, dtype=theano.config.floatX)/127.5 - 1.0

def inverse_transform(X):
    """
    
    Inputs images values are reversed to the [0,255] range
    
    
    :type X: np.array
    :param X: Array of images
    """

    return (X+1.)/2.

def generate_samples(_gen, n_batches, batch_size, size_Z):    
    """
    Generate images using data from a Uniform distrion

    :type n_batches: function
    :param n_batches: Generator function
    
    :type n_batches: int
    :param n_batches: Number of batches to be created
    """
    
    samples = []
    n_gen = 0

    for i in range(n_batches):
        
        random_Z = np.asarray(
                rng.uniform(-1., 1., size=(batch_size, size_Z)),
                dtype=theano.config.floatX)
        random_X = _gen(random_Z)
        samples.append(random_X)

    return np.concatenate(samples, axis=0)