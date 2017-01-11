"""
activations module
"""


def leakyRelu(X, leak=0.2):
    """
    Leeky Relu activation function
    
    :type X: np.array
    :param X: Array of images

    :type leak: float
    :param leak: Decimal value to be leaked
    
    """
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * abs(X)