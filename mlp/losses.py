import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def apply(self, predicted_outputs, expected_outputs):
        raise NotImplementedError()

    @abstractmethod
    def apply_derivative(self, predicted_output, expected_output):
        raise NotImplementedError()


class CrossEntropy(Loss):
    def apply(self, predicted_outputs: list[float], expected_outputs: list[float]):
        epsilon = 1e-8
        # Clip to avoid undefined log(0)
        predicted_outputs = np.clip(predicted_outputs, epsilon, 1 - epsilon)
        return -np.sum(expected_outputs * np.log(predicted_outputs))

    def apply_derivative(self, predicted_output: np.ndarray, expected_output: np.ndarray):
        y = expected_output
        y_hat = predicted_output
        epsilon = 1e-8
        # Clip values to avoid issues with logarithm and division by zero
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        return -(y / y_hat) + (1 - y) / (1 - y_hat)


class MeanSquaredError(Loss):
    def apply(self, predicted_outputs: list[float], expected_outputs: list[float]):
        # cost is sum (for all x,y pairs) of: 0.5 * (x-y)^2
        cost = 0
        for x, y in zip(predicted_outputs, expected_outputs):
            error = x - y
            cost += error * error
        return 0.5 * cost

    def apply_derivative(self, predicted_output: np.ndarray, expected_output: np.ndarray):
        # (0.5 * (x-y)^2)' = x-y
        return predicted_output - expected_output
