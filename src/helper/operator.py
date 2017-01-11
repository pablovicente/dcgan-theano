"""
This code is based on
[1] https://github.com/Newmu/dcgan_code/blob/master/lib/ops.py
"""

import theano
import theano.tensor as T

import numpy as np

from theano.sandbox.cuda.basic_ops import gpu_contiguous, gpu_alloc_empty
from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConvGradI

def drop(input, p=0.7): 
    """
    Set a percentage of the input values to 0. Baseline to implement 
    dropout in the network layers
    
    :type input: numpy.array
    :param input: Layer or weight matrix on which dropout is applied
    
    :type p: float or double between 0. and 1. 
    :param p: Probability of NOT dropping out a unit, therefore (1.-p) is the drop rate.
    
    """            
    rng = np.random.RandomState(1234)
    srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
    return input * mask


def deconv(X, w, subsample=(1, 1), border_mode=(0, 0), conv_mode='conv'):
    """ 
    Deconvolution operation
    
    :type X: np.array
    :param X: Array of real images
    
    :type w: np.array
    :param w: Weight vector
    
    :type subsample: tupe
    :param subsample: Subsample dimension
    
    :type border_mode: string or tuple
    :param border_mode: Border mode to apply
    
    :type conv_mode: string
    :param conv_mode: Type of covolution to appy
        
    """
    img = gpu_contiguous(X)
    kerns = gpu_contiguous(w)
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample,
                          conv_mode=conv_mode)(gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*subsample[0], img.shape[3]*subsample[1]).shape, kerns.shape)
    out = gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*subsample[0], img.shape[3]*subsample[1])
    d_img = GpuDnnConvGradI()(kerns, img, out, desc)
    
    return d_img

def batchnorm(X, gain=None, bias=None, epsilon=1e-8):
    """
    Normalize the tensor to 0 mean and 1 variance
    
    :type X: np.array
    :param X: Array of real images
    
    :type gain: np.array
    :param gain: Gain 1d array 
    
    :type bias: np.array
    :param bias: Bias 1d array 
    
    :type epsilon: np.array
    :param epsilon: Small value to avoid null divisions
        
    """
    
    if X.ndim == 4:
        
        # Reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map.
        b_u = T.mean(X, axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
        b_s = T.mean(T.sqr(X - b_u), axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')

        # Normalize the input array
        X = (X - b_u) / T.sqrt(b_s + epsilon)
        X = X*gain.dimshuffle('x', 0, 'x', 'x') + bias.dimshuffle('x', 0, 'x', 'x')        
    elif X.ndim == 2:
        
        u = T.mean(X, axis=0)
        s = T.mean(T.sqr(X - u), axis=0)

        # Normalize the input array
        X = (X - u) / T.sqrt(s + epsilon)
        X = X*gain + bias       
    else:
        raise NotImplementedError
        
    return X