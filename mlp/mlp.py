import numpy as np
from pydantic import PositiveFloat, PositiveInt

from .layers import Layer, Input, Dense
from .activations import Activation, Linear
from .losses import Loss


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
                 n_epochs: PositiveInt = 10,
                 batch_size: int = 32,
                 learning_rate: PositiveFloat = 0.01,
                 momentum: PositiveFloat = 1.0,  # 1.0 has no effect
                 regularization: PositiveFloat = 1e-3,
                 print_frequency: PositiveInt = 1):
        self.layers = layers
        self.loss_function = loss_function
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.regularization = regularization
        self.print_frequency = print_frequency

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            # Propagate the inputs through the network
            x = layer.forward(x)
        return x

    def backpropagate(self, expected_outputs):
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
            layer.apply_gradients(self.regularization,
                                  self.learning_rate,
                                  self.momentum)

    def fit(self, x: np.ndarray, y: np.ndarray):
        n_samples = x.shape[0]
        for epoch in range(self.n_epochs):
            for batch_start in range(0, n_samples, self.batch_size):
                batch_end = min(batch_start + self.batch_size, n_samples)
                x_batch = x[batch_start:batch_end]
                y_batch = y[batch_start:batch_end]

                # Forward pass
                y_pred = self.forward(x_batch)

                # Backpropagation
                self.backpropagate(y_batch)

            if epoch % self.print_frequency == 0:
                loss = self.loss_function.apply(y_pred, y)
                print(f'Epoch {epoch+1}/{self.n_epochs} | Loss: {loss}')

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
