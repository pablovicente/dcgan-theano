"""
network builder module
"""

import theano.tensor as T

from helper.operator import drop
from helper.activation_function import leakyRelu
from helper.layer import ConvLayer, HiddenLayer, DeConvLayer, LogisticRegression


def build_generator(Z, layer_size, num_filters, filter_size, flat_size, subsamples, 
                    borders, norm, activation, initial_im_size, all_weights=None):
    """
    Creates the generator network from using the provided information

    :type Z: numpy.random.RandomState
    :param Z: 
    
    :type layer_size: list
    :param layer_size: Size of each layer
    
    :type num_filters: list
    :param num_filters: Number of filter per layers
    
    :type filter_size: list
    :param filter_size. Size of each filer
    
    :type flat_size: tuple
    :param flat_size: Size of the first layer
    
    :type subsamples: list
    :param subsamples: Indicates the stride to apply the convolution 
                       operation at each layer
    
    :type borders: list
    :param borders: Indicates the border to apply the convolution 
                    operation at each layer
    
    :type norm: list
    :param norm: Indicates whether to use normalization in each layer
    
    :type activation: list
    :param activation: Activation function to be used
    
    :type all_weights: list
    :param all_weights: Weights to be used in each layer
    """    
    
    # Creates layers are stored to keep track of their parameters
    layers = []    
    weights = None
    
    # First 3 weights (w,g,d) are retrieved from the weight list 
    # in case of providing a set of weights    
    lower = 0
    upper = lower + 3
    if all_weights is not None:                    
        weights = all_weights[lower:upper]
                   
    # Takes random vector as input and performs the dot product between 
    # input and weighs matrix before normalizing
    layer = HiddenLayer(Z, shape=flat_size, layer_index=1, normalization=True,
                        activation=T.nnet.relu, weights=weights)
    
    layers.append(layer)
    layer_input = layer.output
    
    # Layer output is resize to a 4 dimensional tensor in order to perform deconvolutions
    layer_input  = layer_input.reshape((layer_input.shape[0], num_filters[0], initial_im_size, initial_im_size))
    
    for idx in range(len(layer_size)):
        
        # Weight dimension is created
        filter_shape = (num_filters[idx], layer_size[idx], filter_size[idx], filter_size[idx])
        # Either to normalize or not the output
        normalization = norm[idx]
        subsample = subsamples[idx]
        border_mode = borders[idx]

        # Next 3 weights (w,g,d) are retrieved from the weight list 
        # in case of providing a set of weights    
        if all_weights is not None:
            lower = upper
            upper = lower + 3            
            weights = all_weights[lower:upper]
            
        # A layer to perfom decoonvolution is created with the information provided
        layer = DeConvLayer(input=layer_input, filter_shape=filter_shape, subsample=subsample,
                            border_mode=border_mode, layer_index=idx+2, normalization=normalization, 
                            activation=activation[idx], weights=weights)

        # Layer output is the next input, reference to the layer is stored in a list        
        layer_input = layer.output
        layers.append(layer)

    # To make clear that the layer output is an image x
    x = layer.output
    
    return x, layers    



def build_discriminator(X, layer_size, num_filters, filter_size, flat_size, norm, all_weights=None):
    """
    Creates the discriminator network that distinguishes real 
    and generated images

    :type X: 4 dimensional tensor matrix
    :param X: Simbolic variable that describes 
              the input of the discriminator 
    
    :type layer_size: list
    :param layer_size: Size of each layer
    
    :type num_filters: list
    :param num_filters: Number of filter per layers
    
    :type filter_size: list
    :param filter_size. Size of each filer
    
    :type flat_size: tuple
    :param flat_size: Size of the first layer
    
    :type norm: list
    :param norm: Indicates whether to use normalization in each layer
    
    :type activation: list
    :param activation: Activation function to be used
    
    :type all_weights: list
    :param all_weights: Weights to be used in each layer
    """  
    
    # Creates layers are stored to keep track of their parameters    
    layers = []
    weights = None
        
    layer_input = X
    upper = 1
    for idx in range(len(layer_size)):
        
        # Weight dimension is created        
        filter_shape = (num_filters[idx], layer_size[idx], filter_size[idx], filter_size[idx])
        # Either to normalize or not the output        
        normalization = norm[idx]
        
        # Next 3 weights (w,g,d) are retrieved from the weight list 
        # in case of providing a set of weights            
        if all_weights is not None:
            if idx == 0:
                weights = [all_weights[idx]]
            else:
                lower = upper
                upper = lower + 3
                weights = all_weights[lower:upper]
                    
        # A layer to perfom convolution is created with the information provided            
        layer = ConvLayer(input=layer_input, weights=weights, filter_shape=filter_shape, subsample=(2,2), 
                          layer_index=idx+1, normalization=normalization, activation=leakyRelu)

        
        # Layer output is the next input, reference to the layer is stored in a list
        layer_input = drop(layer.output, 0.7)
        layers.append(layer)


    if all_weights is not None:
        weights = all_weights[-1]
        
    # The convolutional layer output is flatten and pass to a layer
    # that uses sigmoid activation    
    layer_input = T.flatten(layer_input, 2)
    
    layer = LogisticRegression(input=layer_input, shape=flat_size, weights=weights)
    layers.append(layer)
    
    # To make clear is the target value (y)
    y = layer.output
        
    return y, layers