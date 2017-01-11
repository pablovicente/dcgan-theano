import theano
import theano.tensor as T

import numpy as np

class Adam(object):
    """
    Adam optimizer    
    """
        
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, l2=0.0, e=1e-8):
        """ 
        :type lr: numpy.array
        :param lr: Learning rate
                
        :type b1: float
        :param b1: Beta 1 parameter 
        
        :type b2: float
        :param b2: Beta 2 parameter
        
        :type l2: float
        :param l2: L2 regularization        
        
        :type e: float
        :param e: Epsilon value
        """       
        
        self.__dict__.update(locals())  

    def __call__(self, params, cost):
        """
        :type params: list
        :param params: Parameters to be updated
        
        :type cost: list
        :param cost: Cost to be optimized
        """
        updates = []
        grads = T.grad(cost, params)
        t = theano.shared(np.asarray(1., dtype=theano.config.floatX))
        b1_t = self.b1*self.e**(t-1)
     
        for p, g in zip(params, grads):
            # Regularize gradient
            g += p * self.l2

            m = theano.shared(p.get_value() * 0.)
            v = theano.shared(p.get_value() * 0.)
     
            m_t = b1_t*m + (1 - b1_t)*g
            v_t = self.b2*v + (1 - self.b2)*g**2
            m_c = m_t / (1-self.b1**t)
            v_c = v_t / (1-self.b2**t)
            p_t = p - (self.lr * m_c) / (T.sqrt(v_c) + self.e)
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t) )
            
        updates.append((t, t + 1.))
        return updates
    
        
class SGD(object):
    """
    Stochastic Gradient Descent optimizer
    """
    
    def __init__(self, learning_rate):
        self.lr = learning_rate
        pass

    def __call__(self, params, cost):
        """
        :type params: list
        :param params: Parameters to be updated
        
        :type cost: list
        :param cost: Cost to be optimized
        """

        updates = []        
        for param in params:
            updates.append((param, param - self.lr * T.grad(cost, param)))
            
        return updates    
    