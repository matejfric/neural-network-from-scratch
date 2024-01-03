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
    def apply(self, z):
        return 1.0/(1.0+np.exp(-z))

    def apply_derivative(self, z):
        return self.apply(z)*(1-self.apply(z))


class ReLU(Activation):
    def apply(self, z):
        return np.maximum(0, z)

    def apply_derivative(self, z):
        return np.heaviside(z, 1/2)


class Softmax(Activation):
    def apply(self, z):
        # prevent overflow
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)  # probabilities
