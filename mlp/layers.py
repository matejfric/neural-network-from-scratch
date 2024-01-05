import numpy as np
from pydantic import PositiveFloat, PositiveInt
from abc import ABC, abstractmethod
import logging

from .activations import Activation
from .constants import SEED
from .optimizers import Optimizer


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
        self._weight_m = None
        self._bias_m = None
        self._weight_v = None
        self._bias_v = None

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
        self._weight_m = np.zeros_like(self._weights)  # 1st stat. moment cache
        self._bias_m = np.zeros_like(self._biases)  # 1st stat. moment cache
        self._weight_v = np.zeros_like(self._weights)  # 2nd stat. moment cache
        self._bias_v = np.zeros_like(self._biases)  # 2nd stat. moment cache

        # For RMSProp
        # self._weight_v = np.ones_like(self._weights)
        # self._bias_v = np.ones_like(self._biases)

    def forward(self, inputs: np.ndarray):
        self._inputs = inputs
        self._weighted_inputs = np.dot(inputs, self._weights) + self._biases
        self._activations = self.activation.apply(self._weighted_inputs)
        # self.output = np.dot(inputs, self.weights) + self.biases
        # self.output = self.activation.activate(self.output)
        return self._activations  # self.output

    def apply_gradients(self,
                        optimizer: Optimizer,
                        regularization: PositiveFloat,
                        learning_rate: PositiveFloat,
                        momentum: PositiveFloat):
        optimizer_functions = {
            Optimizer.SGD: self._sgd,
            Optimizer.SGD_MOMENTUM: self._sgd_momentum,
            Optimizer.ADAGRAD: self._adagrad,
            Optimizer.RMSPROP: self._rmsprop,
            Optimizer.ADAM: self._adam,
        }
        # Call the function corresponding to the optimizer
        optimize = optimizer_functions.get(optimizer,
                                lambda: logging.warning("Invalid optimizer"))
        optimize(regularization, learning_rate, momentum)

    def _sgd(self, 
             regularization: PositiveFloat,
             learning_rate: PositiveFloat,
             momentum: PositiveFloat):
        """Stochastic gradient descent (SGD)"""
        self._weights -= self.gradient_w * learning_rate
        self._biases -= self.gradient_b * learning_rate

    def _sgd_momentum(self,
                      regularization: PositiveFloat,
                      learning_rate: PositiveFloat,
                      momentum: PositiveFloat):
        """
        Stochastic gradient descent (with momentum),
        if momentum=0, then this becomes regular SGD.
        """
        gw = self.gradient_w  # gradient for weights 
        vw = self._weight_m  # velocities for weights
        self._weight_m = momentum * vw - learning_rate * gw 
        self._weights += self._weight_m

        gb = self.gradient_b  # gradient for biases 
        vb = self._bias_m  # velocities for biases
        self._bias_m = momentum * vb - learning_rate * gb
        self._biases += self._bias_m

    def _adagrad(self,
                 regularization: PositiveFloat,
                 learning_rate: PositiveFloat,
                 momentum: PositiveFloat,
                 epsilon: PositiveFloat = 1e-8):
        """ ADAGrad """
        gw = self.gradient_w  # gradient for weights 
        self._weight_v += np.square(gw)  # cache of squared gradients
        self._weights -= learning_rate * gw / np.sqrt(self._weight_v + epsilon)

        gb = self.gradient_b  # gradient for biases 
        self._bias_v += np.square(gb)  # cache of squared gradients
        self._biases -= learning_rate * gb / np.sqrt(self._bias_v + epsilon)

    def _rmsprop(self,
                 regularization: PositiveFloat,
                 learning_rate: PositiveFloat,
                 momentum: PositiveFloat,
                 decay: PositiveFloat = 0.999,
                 epsilon: PositiveFloat = 1e-8):
        """ Root Mean Square Propagation (RMSProp) """
        gw = self.gradient_w  # gradient for weights 
        vw = self._weight_v  # cache of squared gradients
        self._weight_v = decay * vw + (1-decay) * np.square(gw)
        self._weights -= learning_rate * gw / np.sqrt(self._weight_v + epsilon)

        gb = self.gradient_b  # gradient for biases 
        vb = self._bias_v  # cache of squared gradients
        self._bias_v = decay * vb + (1-decay) * np.square(gb)
        self._biases -= learning_rate * gb / np.sqrt(self._bias_v + epsilon)

    def _adam(self,
                 regularization: PositiveFloat,
                 learning_rate: PositiveFloat = 0.001,
                 momentum: PositiveFloat = 0.9,
                 decay: PositiveFloat = 0.999,
                 epsilon: PositiveFloat = 1e-8):
        """ Adaptive Momentum Estimator (Adam) """
        gw = self.gradient_w  # gradient for weights 
        mw = self._weight_m  # cache of gradients
        vw = self._weight_v  # cache of squared gradients

        # Estimate 1st stat. moment
        self._weight_m = momentum * mw + (1-momentum) * gw
        self._weight_m /= (1-momentum)

        # Estimate 2nd stat. moment
        self._weight_v = decay * vw + (1-decay) * np.square(gw)
        self._weight_v /= (1-decay)

        # Update weigths
        self._weights -= learning_rate * self._weight_m / np.sqrt(self._weight_v + epsilon)

        # Likewise for biases
        gb = self.gradient_b  # gradient for biases 
        mb = self._bias_m  # cache of gradients
        vb = self._bias_v  # cache of squared gradients

        # Estimate 1st stat. moment
        self._bias_m = momentum * mb + (1-momentum) * gb
        self._bias_m /= (1-momentum)

        # Estimate 2nd stat. moment
        self._bias_v = decay * vb + (1-decay) * np.square(gb)
        self._bias_v /= (1-decay)

        # Update biases
        self._biases -= learning_rate * self._bias_m / np.sqrt(self._bias_v + epsilon)

    # TODO:
    def backward(self, inputs: np.ndarray):
        return super().backward(inputs)
