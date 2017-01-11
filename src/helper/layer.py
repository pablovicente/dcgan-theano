"""
Layer classes

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
"""

import numpy
import theano
import theano.tensor as T

from helper.initialization import normal, constant
from helper.operator import deconv, batchnorm
from helper.activation_function import leakyRelu

from theano.sandbox.cuda.dnn import dnn_conv


class HiddenLayer(object):
    """Fully connected hidden layer """

    def __init__(self,
                input,
                shape,
                layer_index,
                normalization=True, 
                activation=T.nnet.relu,
                weights=None):
        """
        Allocate a HiddenLayer with shared variable internal parameters.

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type shape: tuple or list of length 4
        :param shape: (size of random vector, num filters*8*4*4)

        :type layer_index: int
        :param layer_index: Layer index used when creating the name
        
        :type normalization: boolean
        :param normalization: Indicates whether to normalized output or not
        
        :type activation: activation function
        :param activation: Transfer function used in the layer
        
        :type weights: list
        :param weights: list of weighs for each layer parameter
        """


        self.input = input

        if weights is None:
            self.w = normal(loc=0.0, scale=0.02, shape=shape, name='w'+str(layer_index)) 
            self.g = normal(loc=1.0, scale=0.02, shape=shape[1], name='g'+str(layer_index))
            self.b = constant(value=0.0, shape=shape[1], name='b'+str(layer_index))
        else:
            self.w, self.g, self.b = weights

            
        self.output = T.dot(input, self.w)

        if normalization:
            self.output = batchnorm(self.output, gain=self.g, bias=self.b)

        self.output = activation(self.output)

        # store parameters of this layer
        self.params = [self.w]
        if normalization:
            self.params += [self.g, self.b]
        
class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, shape, weights=None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type shape: tuple
        :param shape: (num_filter*8*4*4, 1)
        
        :type weights: list
        :param weights: list of weights for each layer parameter

        """
        if weights is None:
            self.dwy = normal(loc=0.0, scale=0.02, shape=shape, name='dwy')
        else:
            self.dwy = weights

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.output = T.nnet.sigmoid(T.dot(input, self.dwy))

        # parameters of the model
        self.params = [self.dwy]

        # keep track of model input
        self.input = input
    
class ConvLayer(object):
    """Conv Layer of a convolutional network """

    def __init__(self,
                input,
                filter_shape,
                subsample, 
                layer_index,
                normalization=True, 
                activation=T.nnet.relu,
                weights=None):
        """
        Allocate a ConvLayer with shared variable internal parameters.

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type shape: tuple or list of length 4
        :param shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type subsample: tuple or list of length 4
        :param subsample: (size, size) (2,2)
                             
        :type layer_index: int
        :param layer_index: Layer index used when creating the name
        
        :type normalization: boolean
        :param normalization: Indicates whether to normalized output or not
        
        :type activation: activation function
        :param activation: Transfer function used in the layer
        
        :type weights: numpy.random.RandomState
        :param weights: 
        """
        self.input = input
        
        # Weights are created if they are not passed as a parameter
        if weights is None:
            self.dw = normal(loc=0.0, scale=0.02, shape=filter_shape, name='dw'+str(layer_index))             
            self.dg = normal(loc=1.0, scale=0.02, shape=filter_shape[0], name='dg'+str(layer_index))
            self.db = constant(value=0.0, shape=filter_shape[0], name='db'+str(layer_index))
        else:
            # If normalization is being used, 3 different parameters are passed
            # Otherwise only one (weight matrix)            
            if normalization:
                self.dw, self.dg, self.db = weights
            else:
                self.dw = weights[0]

        # Perform convolution between input and weight matrix
        self.output = dnn_conv(self.input, self.dw, subsample=(2, 2), border_mode=(2, 2))

        # If batch normalization is being used, it is applied to the output         
        if normalization:
            self.output = batchnorm(self.output, gain=self.dg, bias=self.db)

        # The output is the pass through the activation function
        self.output = activation(self.output)

        # Number of parameters depend on whether normalization is being used
        self.params = [self.dw] 
        if normalization:
            self.params += [self.dg, self.db]

class DeConvLayer(object):
    """Conv Layer of a convolutional network """

    def __init__(self,
                input,
                filter_shape,
                subsample,
                border_mode,
                layer_index,
                normalization=True, 
                activation=leakyRelu,
                weights=None):
        """
        Allocate a DeConvLayer with shared variable internal parameters.

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)
                             
        :type layer_index: int
        :param layer_index: Layer index used when creating the name
        
        :type normalization: boolean
        :param normalization: Indicates whether to normalized output or not
        
        :type activation: activation function
        :param activation: Transfer function used in the layer
        
        :type weights: list
        :param weights: list of weights for each layer parameter
        """

        self.input = input
        
        # Weights are created if they are not passed
        if weights is None:
            self.gw = normal(loc=0.0, scale=0.02, shape=filter_shape, name='gw'+str(layer_index))
            self.gg = normal(loc=1.0, scale=0.02, shape=filter_shape[1], name='gg'+str(layer_index))
            self.gb = constant(value=0.0, shape=filter_shape[1], name='gb'+str(layer_index))
        else:
            # If normalization is being used, 3 different parameters are passed.
            # Otherwise only one (weight)
            if normalization:
                self.gw, self.gg, self.gb = weights
            else:
                self.gw = weights[0]
                
        # Perform deconvolution between input and weight matrix        
        self.output = deconv(self.input, self.gw, subsample=subsample, border_mode=subsample)

        # If batch normalization is being used, it is applied to the output 
        if normalization:
            self.output = batchnorm(self.output, gain=self.gg, bias=self.gb)

        # The output is the pass through the activation function            
        self.output = activation(self.output)

        # Number of parameters depend on whether normalization is being used
        self.params = [self.gw] 
        if normalization:
            self.params += [self.gg, self.gb]