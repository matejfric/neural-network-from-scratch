import numpy as np
from abc import ABC, abstractmethod

# def categorical_cross_entropy(y_pred, y_true):
#     """
#     Args:
#         y_pred - output of the softmax function
#         y_true - one-hot encoded class targets
#     """
#     samples = len(y_pred)
#     y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) # logarithm infinity

#     if len(y_true.shape) == 1:
#         correct_confidences = y_pred_clipped[range(samples), y_true]

#     elif len(y_true.shape) == 2:
#         correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

#     negative_log_likelihoods = -np.log(correct_confidences)
#     return negative_log_likelihoods


class Loss(ABC):
    @abstractmethod
    def apply(self, predicted_outputs, expected_outputs):
        raise NotImplementedError()

    @abstractmethod
    def apply_derivative(self, predicted_output, expected_output):
        raise NotImplementedError()


class CrossEntropy(Loss):
    def apply(self, predicted_outputs: list[float], expected_outputs: list[float]):
        # cost is sum (for all x,y pairs) of: -y * log(x) - (1-y) * log(1-x)
        cost = 0
        for x, y in zip(predicted_outputs, expected_outputs):
            v = -np.log(x) if y == 1 else -np.log(1 - x)
            cost += 0 if np.isnan(v) else v
        return cost

    # def apply_derivative(self, predicted_output: float, expected_output: float):
    #     x = predicted_output
    #     y = expected_output
    #     if x == 0 or x == 1:
    #         return 0
    #     return (-x + y) / (x * (x - 1))

    def apply_derivative(self, predicted_output: np.ndarray, expected_output: np.ndarray):
        x = predicted_output
        y = expected_output

        mask = (x != 0) & (x != 1)
        result = np.zeros_like(x, dtype=float)

        result[mask] = (-x[mask] + y[mask]) / (x[mask] * (x[mask] - 1))

        return result


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
