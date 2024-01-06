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
        # https://mattpetersen.github.io/softmax-with-cross-entropy
        # 'inputs' has shape (batch_size, output.shape[1])
        # compute batch jacobian (assuming OHE vector):
        # --> np.diag(np.diagflat(s) - np.outer(s, s)).reshape(s.shape)
        # jacobian:
        # np.diagflat(s) - np.outer(s, s)
        softmax_output = self.apply(inputs)
        return softmax_output * (1 - softmax_output)


class Linear(Activation):
    def apply(self, inputs):
        return inputs
    
    def apply_derivative(self, inputs):
        return np.ones_like(inputs)
