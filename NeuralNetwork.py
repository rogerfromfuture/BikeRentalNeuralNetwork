import numpy as np


def MSE(y, Y):
    return np.mean((y - Y) ** 2)


# This NeuralNetwork weight is define as from input -> output direction
class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes ** -0.5,
                                                        (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes ** -0.5,
                                                         self.hidden_nodes)
        self.lr = learning_rate

        # Activation function is the sigmoid function
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

    def train(self, inputs_list, targets_list):
        inputs = inputs_list
        targets = targets_list

        # Implement the forward pass here
        # Forward pass ###
        # Hidden layer
        hidden_inputs = np.dot(inputs, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        # Output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs

        # Implement the backward pass here
        # Backward pass

        # Output error
        # Calculate error gradient in output units
        output_errors = targets - final_outputs

        # Calculate error gradient in hidden units
        hidden_errors = np.dot(self.weights_hidden_to_output, output_errors)
        hidden_grad = hidden_outputs * (1 - hidden_outputs)

        # Update the weights
        self.weights_hidden_to_output += self.lr * output_errors * hidden_outputs
        self.weights_input_to_hidden += self.lr * hidden_errors * hidden_grad * inputs[:, None]

    def run(self, inputs_list):
        # Run a forward pass through the network
        inputs = inputs_list

        # Implement the forward pass here
        # Hidden layer
        hidden_inputs = np.dot(inputs, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        # Output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs

        return final_outputs
