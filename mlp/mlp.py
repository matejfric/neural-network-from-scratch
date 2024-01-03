import numpy as np
from pydantic import PositiveFloat, PositiveInt

from .layers import Layer, Input, Dense
from .activations import Activation
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
                 learning_rate: PositiveFloat = 0.01,
                 momentum: PositiveFloat = 1.0,  # 1.0 has no effect
                 regularization: PositiveFloat = 1e-3):
        self.layers = layers
        self.loss_function = loss_function
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.regularization = regularization

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            # Propagate the inputs through the network
            x = layer.forward(x)
        # for layer in self.layers:
        #     layer.backward(x)
        return x

    def backpropagate(self, expected_outputs):
        """
        batch size agnostic
        """

        # OUTPUT LAYER
        output_layer = self.layers[-1]
        # for i in range(output_layer.n_neurons):
        # ∂(loss) / ∂(activation) ...pd of cost wrt output of the last layer

        # compute loss
        # print(self.loss_function.apply(output_layer.activations, expected_outputs))

        loss_derivative = self.loss_function.apply_derivative(
            output_layer.activations, expected_outputs
        )
        # ∂(activation) / ∂(weighted inputs)
        activation_derivative = output_layer.activation.apply_derivative(
            output_layer.weighted_inputs
        )
        output_delta = loss_derivative * activation_derivative
        output_pd = self.layers[-2].activations[:, :,
                                                np.newaxis] * output_delta[:, np.newaxis, :]
        # ∂(cost) / ∂(weights)
        # average over batch dimension (data points)
        output_layer.gradient_w = np.mean(output_pd, axis=0)
        # ∂(cost) / ∂(biases)
        # now (1, n_neurons); may be transposed to make it a column vector
        output_layer.gradient_b = 1 * \
            np.mean(output_delta, axis=0)[np.newaxis, :]

        hidden_layer = self.layers[-2]
        activation_derivative = hidden_layer.activation.apply_derivative(
            hidden_layer.weighted_inputs)  # activations?

        layer_delta = activation_derivative * \
            np.dot(output_delta, output_layer._weights.T)
        layer_pd = self.layers[-3].activations[:, :,
                                               np.newaxis] * layer_delta[:, np.newaxis, :]

        # Update gradients for weights and biases
        hidden_layer.gradient_w = np.mean(layer_pd, axis=0)  # this is not fine
        hidden_layer.gradient_b = 1 * \
            np.mean(layer_delta, axis=0)[np.newaxis, :]

        # BACKPROPAGATION THROUGH HIDDEN LAYERS
        # layer_delta = output_delta
        # for i in range(len(self.layers) - 2, 0, -1):
        #     layer_delta = self._update_hidden_gradients(self.layers[i], self.layers[i+1], layer_delta)
        # for i, layer in enumerate(reversed(self.layers[:-1])):
        #    self._update_hidden_gradients(layer, self.layers[-(i+1)]) # TODO last argument

        # GRADIENT DESCENT - UPDATE WEIGHTS AND BIASES
        for layer in self.layers[1:]:
            # Except input layer
            layer.apply_gradients(self.regularization,
                                  self.learning_rate, self.momentum)

    def fit(self, x, y):
        for epoch in range(self.n_epochs):
            # for batch in range(x.shape[0])
            y_pred = self.forward(x)
            self.backpropagate(y)

            if epoch % 1000 == 0:
                loss = self.loss_function.apply(y_pred, y)
                print(f'Epoch {epoch+1}/{self.n_epochs} | Loss: {loss}')

    def _update_hidden_gradients(self, layer: Layer, next_layer: Layer, next_layer_delta):
        # Compute the delta for the hidden layer
        activation_derivative = layer.activation.apply_derivative(
            layer.weighted_inputs)  # activations?

        # sum over next_layer.n_neurons
        # layer_delta = []
        # for n in range(next_layer.n_neurons):
        #     layer_delta.append(next_layer._weights)

        layer_delta = activation_derivative * \
            np.dot(next_layer_delta, next_layer._weights.T)
        layer_pd = layer.activations[:, :,
                                     np.newaxis] * layer_delta[:, np.newaxis, :]

        # Update gradients for weights and biases
        layer.gradient_w = np.mean(layer_pd, axis=0)  # this is not fine

        layer.gradient_b = np.sum(layer_delta, axis=0, keepdims=True)

    # def _update_hidden_gradients(self,
    #                              layer: Layer,
    #                              prev_layer: Layer,
    #                              prev_layer_delta = None):
    #     # ∂(weighted inputs^(prev_layer)) / ∂(activation) = prev_layer.weights
    #     weighted_input_derivative = prev_layer._weights
    #     # backward propagation of error
    #     layer_delta = weighted_input_derivative * prev_layer.gradient_w #prev_layer_delta # the code fails here
    #     # ans * ∂(activation) / ∂(weighted inputs)
    #     layer.gradient_w = layer_delta * np.sum(layer.activation.apply_derivative(
    #         layer.weighted_inputs), axis=0)
    #     layer.gradient_b = 1 * layer_delta

        # layer.gradient_b = 1 * layer.gradient_w

    def __str__(self):
        layer_info = [
            f"Layer {i}: {type(layer).__name__} - {layer.n_neurons} neurons" for i, layer in enumerate(self.layers)]
        return "\n".join(layer_info)


if __name__ == "__main__":
    from activations import Sigmoid
    from losses import MeanSquaredError

    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    y = np.array([[0],
                  [1],
                  [1],
                  [0]])

    # y = np.array([[0,1],
    #             [1,0],
    #             [1,0],
    #             [0,0]])

    mlp_layers_builder = MLPLayersBuilder()\
        .add_input(2)\
        .add_dense(5, Sigmoid())\
        .add_dense(1, Sigmoid())  # .add_dense(3, Sigmoid())\
    layers = mlp_layers_builder.build()
    mlp = MLP(layers, MeanSquaredError(), 10000, learning_rate=0.9)

    print(mlp.forward(X))

    mlp.fit(X, y)

    print(mlp.forward(X))
