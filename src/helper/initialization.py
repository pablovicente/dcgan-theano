"""
initialization module
"""

import numpy
import theano

def normal(loc, scale, shape, shared=True, name=None):
    """
    Create a theano shared variables with values from a Normal distribution    

    :type loc: float
    :param loc: Mean of the distribution

    :type scale: int
    :param scale: Standard deviation of the distribution

    :type shape: tuple
    :param shape: Size of the array to be created
    
    :type shared: boolean
    :param shared: Whether to make it a share variable

    :type name: string
    :param name: Name of the variable created
    """
    
    
    X = numpy.asarray(
        numpy.random.normal(loc=loc, scale=scale, size=shape),
        dtype=theano.config.floatX
    )

    if shared:
        return theano.shared(value=X, name=name, borrow=True)
    else:
        return X
    

def uniform(low, high, shape, shared=True, name=None):
    """
    Create a theano shared variables with values from a Uniform distribution    

    :type low: float
    :param low: Lower boundary for Uniform distribution

    :type high: float
    :param high: Higher boundary for Uniform distribution

    :type shape: tuple
    :param shape: Size of the array to be created
    
    :type shared: boolean
    :param shared: Whether to make it a share variable

    :type name: string
    :param name: Name of the variable created
    """

    X = numpy.asarray(
        numpy.random.uniform(low=low, high=high, size=shape),
        dtype=theano.config.floatX
    )

    if shared:
        return theano.shared(value=X, name=name, borrow=True)
    else:
        return X

def constant(value, shape, shared=True, name=None):
    """
    Create a theano shared variables with values from a Normal distribution    

    :type value: float
    :param value: Value of the variable to be created
    
    :type shape: tuple
    :param shape: Size of the array to be created
    
    :type shared: boolean
    :param shared: Whether to make it a share variable

    :type name: string
    :param name: Name of the variable created
    """

    X = numpy.ones(shape, dtype=theano.config.floatX)*value

    if shared:
        return theano.shared(value=X, name=name, borrow=True)
    else:
        return X

