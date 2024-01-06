import numpy as np
from typing import Optional
from pydantic import PositiveFloat, PositiveInt

from .layers import Layer, Input, Dense
from .activations import Activation, Linear
from .losses import Loss
from .constants import SEED
from .optimizers import Optimizer


class MLPLayersBuilder:
    """
    Builder for MLP
    """

    def __init__(self):
        self.layers: list[Layer] = []

    def add_input(self, n_inputs):
        input_layer = Input(n_inputs)
        self.layers.append(input_layer)
        return self

    def add_dense(self, n_neurons, activation: Activation):
        dense_layer = Dense(n_neurons, activation)
        self.layers.append(dense_layer)
        return self

    def build(self):
        if len(self.layers) < 2:
            raise Exception(
                'The MLP must have at least an input layer and output layer!')
        if not isinstance(self.layers[0], Input):
            raise Exception('The input layer is missing!')

        self.layers[0].activation = Linear()
        input_size = self.layers[0].n_neurons
        for layer in self.layers[1:]:
            layer.set_input(input_size)
            input_size = layer.n_neurons

        return self.layers


class MLP:
    def __init__(self,
                 layers: list[Layer],
                 loss_function: Loss,
                 optimizer: Optimizer = Optimizer.SGD,
                 n_epochs: PositiveInt = 10,
                 batch_size: int = 32,
                 learning_rate: PositiveFloat = 0.01,
                 momentum: PositiveFloat = 0.0,  # 0.0 has no effect
                 print_frequency: PositiveInt = 1,
                 shuffle: bool = False):
        self.layers = layers
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.print_frequency = print_frequency
        self.shuffle = shuffle
        self.rng = np.random.default_rng(SEED)

    def _forward(self, inputs):
        x = inputs
        for layer in self.layers:
            # Propagate the inputs through the network
            x = layer.forward(x)
        return x

    def _backpropagate(self, expected_outputs):
        """
        Batch size agnostic
        """

        # OUTPUT LAYER ERROR
        output_layer = self.layers[-1]

        # ∂(loss) / ∂(activation) ...pd of cost wrt output of the last layer
        loss_derivative = self.loss_function.apply_derivative(
            output_layer.activations, expected_outputs
        )
        # ∂(activation) / ∂(weighted inputs)
        activation_derivative = output_layer.activation.apply_derivative(
            output_layer.weighted_inputs
        )
        output_delta = loss_derivative * activation_derivative
        # ∂(cost) / ∂(weights)
        output_pd = self.layers[-2].activations[:, :, np.newaxis] *\
              output_delta[:, np.newaxis, :]
        # average over batch dimension (data points)
        output_layer.gradient_w = np.mean(output_pd, axis=0)
        # ∂(cost) / ∂(biases)
        # shape (1, n_neurons); may be transposed to make it a column vector
        output_layer.gradient_b = 1 * \
            np.mean(output_delta, axis=0)[np.newaxis, :]

        # BACKPROPAGATION THROUGH HIDDEN LAYERS
        layer_delta = output_delta
        # Loop over hidden layers (in reversed order)
        for hidden_layer_index in range(len(self.layers) - 2, 0, -1):
            layer_delta = self._update_hidden_gradients(hidden_layer_index, layer_delta)

        # GRADIENT DESCENT - UPDATE WEIGHTS AND BIASES
        for layer in self.layers[1:]:
            # Except input layer
            layer.apply_gradients(self.optimizer,
                                  self.learning_rate,
                                  self.momentum)

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            x_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None):
        n_samples = x.shape[0]
        for epoch in range(self.n_epochs):
            if self.shuffle:
                # Shuffle data at the beginning of each epoch
                indices = self.rng.permutation(n_samples)
                x = x[indices]
                y = y[indices]

            for batch_start in range(0, n_samples, self.batch_size):
                batch_end = min(batch_start + self.batch_size, n_samples)
                x_batch = x[batch_start:batch_end]
                y_batch = y[batch_start:batch_end]

                # Forward pass
                y_pred = self._forward(x_batch)

                # Backpropagation
                self._backpropagate(y_batch)

            if epoch % self.print_frequency == 0:
                if x_val is not None and y_val is not None:
                    y_pred = self.predict(x_val)
                    loss = self.loss_function.apply(y_pred, y_val)
                else:
                    y_pred = self.predict(x)
                    loss = self.loss_function.apply(y_pred, y)
                print(f'Epoch {epoch+1}/{self.n_epochs} | Loss: {loss}')

    def predict(self, x: np.ndarray) -> np.ndarray:
        n_samples = x.shape[0]
        predictions = []
        for batch_start in range(0, n_samples, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_samples)
            x_batch = x[batch_start:batch_end]

            # Forward pass
            y_pred = self._forward(x_batch)
            predictions.append(y_pred)
        return np.concatenate(predictions, axis=0)

    def _update_hidden_gradients(self,
                                 layer_index: PositiveInt,
                                 next_layer_delta: np.ndarray):
        prev_layer = self.layers[layer_index - 1]
        hidden_layer = self.layers[layer_index]
        next_layer = self.layers[layer_index + 1]

        activation_derivative = hidden_layer.activation.apply_derivative(
            hidden_layer.weighted_inputs)
        layer_delta = activation_derivative * \
            np.dot(next_layer_delta, next_layer._weights.T)
        layer_pd = prev_layer.activations[:, :,np.newaxis]\
              * layer_delta[:, np.newaxis, :]

        # Update gradients for weights and biases
        hidden_layer.gradient_w = np.mean(layer_pd, axis=0)
        hidden_layer.gradient_b = np.mean(layer_delta, axis=0)[np.newaxis, :]
        
        return layer_delta

    def __str__(self):
        title = ["Multi-layer Perceptron"]
        layer_info = [f" - Layer {i}: {type(layer).__name__} with "
                      f"{type(layer.activation).__name__} activation and "
                      f"{layer.n_neurons} neurons"
                      for i, layer in enumerate(self.layers)]
        return "\n".join(title + layer_info)
