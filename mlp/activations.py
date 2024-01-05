import numpy as np
from enum import Enum, auto
from abc import ABC, abstractmethod


class Activation(ABC):
    @abstractmethod
    def apply(self, inputs: np.ndarray):
        raise NotImplementedError()

    @abstractmethod
    def apply_derivative(self, inputs: np.ndarray):
        raise NotImplementedError()


class Sigmoid(Activation):
    def apply(self, inputs):
        return 1.0/(1.0+np.exp(-inputs))

    def apply_derivative(self, inputs):
        return self.apply(inputs)*(1-self.apply(inputs))


class ReLU(Activation):
    def apply(self, inputs):
        return np.maximum(0, inputs)

    def apply_derivative(self, inputs):
        return np.heaviside(inputs, 1/2)
    

class Softmax(Activation):
    def apply(self, inputs: np.ndarray):
        # prevent overflow
        exp_inputs = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        return exp_inputs / np.sum(exp_inputs, axis=-1, keepdims=True)

    def apply_derivative(self, inputs: np.ndarray):
        # https://stackoverflow.com/a/40576872
        # https://medium.com/intuitionmath/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
        # 'inputs' has shape (1,n)
        # Reshape the 1-d 'inputs' to 2-d so that np.dot 
        # will do the matrix multiplication
        s = inputs.reshape((-1,1))
        jacobian = np.diagflat(s) - np.dot(s, s.T) # (n, n) matrix
        return jacobian


class Linear(Activation):
    def apply(self, inputs):
        return inputs
    
    def apply_derivative(self, inputs):
        return np.ones_like(inputs)
