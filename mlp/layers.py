import numpy as np
from pydantic import PositiveFloat, PositiveInt
from abc import ABC, abstractmethod

from .activations import Activation
from .constants import SEED


class Layer(ABC):
    def __init__(self, n_neurons: PositiveInt, *args, **kwargs):
        self.rng = np.random.default_rng(SEED)
        self.n_neurons: PositiveInt = n_neurons  # #nodes_out
        self.activation: Activation = None
        self._weights = None  # Matrix (#nodes_in, #nodes_out)
        self._biases = None  # Matrix (1, #n_neurons)
        self._gradient_w = None  # cost_gradient_w
        self._gradient_b = None  # cost_gradient_b
        self._inputs = None
        self._weighted_inputs = None
        self._activations = None
        self._weight_velocities = None
        self._bias_velocities = None

    @property
    def gradient_w(self):
        return self._gradient_w

    @gradient_w.setter
    def gradient_w(self, value):
        self._gradient_w = value

    @property
    def gradient_b(self):
        return self._gradient_b

    @gradient_b.setter
    def gradient_b(self, value):
        self._gradient_b = value

    @property
    def inputs(self):
        return self._inputs

    @property
    def weighted_inputs(self):
        return self._weighted_inputs

    @property
    def activations(self):
        return self._activations

    @abstractmethod
    def forward(self, inputs: np.ndarray):
        raise NotImplementedError()

    @abstractmethod
    def backward(self, inputs: np.ndarray):
        raise NotImplementedError()


class Input(Layer):
    def __init__(self, n_neurons):
        super().__init__(n_neurons)

    def forward(self, inputs):
        self._activations = inputs
        return inputs

    def backward(self, inputs):
        return inputs


class Dense(Layer):
    def __init__(self, n_neurons: PositiveInt, activation: Activation):
        super().__init__(n_neurons)
        self.activation = activation
        self._weights = None
        self._biases = None
        # self.output = None

    def set_input(self, n_inputs: PositiveInt):
        self._weights = 0.10 * \
            self.rng.standard_normal((n_inputs, self.n_neurons))
        self._biases = np.zeros((1, self.n_neurons))
        self._weight_velocities = np.zeros_like(self._weights)
        self._bias_velocities = np.zeros_like(self._biases)

    def forward(self, inputs: np.ndarray):
        self._inputs = inputs
        self._weighted_inputs = np.dot(inputs, self._weights) + self._biases
        self._activations = self.activation.apply(self._weighted_inputs)
        # self.output = np.dot(inputs, self.weights) + self.biases
        # self.output = self.activation.activate(self.output)
        return self._activations  # self.output

    def apply_gradients(self,
                        regularization: PositiveFloat,
                        learning_rate: PositiveFloat,
                        momentum: PositiveFloat):
        # Simplified: layer.weights -= self.LEARNING_RATE * pd_error_wrt_weight

        # weight_decay = (1 - regularization * learning_rate)

        # Update weights
        # self._weight_velocities *= momentum - self.gradient_w * learning_rate
        # self._weights *= weight_decay + self._weight_velocities
        self._weights -= self.gradient_w * learning_rate

        # Update biases
        # self._bias_velocities *=  momentum - self.gradient_b * learning_rate
        # self._biases += self._bias_velocities
        self._biases -= self.gradient_b * learning_rate

    # TODO:
    def backward(self, inputs: np.ndarray):
        return super().backward(inputs)
