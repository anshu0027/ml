import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5):
        # Initialize weights and biases for input-hidden and hidden-output layers
        self.weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))
        self.bias_hidden = np.random.uniform(size=(1, hidden_size))
        self.bias_output = np.random.uniform(size=(1, output_size))
        self.learning_rate = learning_rate

    def feedforward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)

        # Calculate final output
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)

        return self.final_output

    def backpropagation(self, X, y, output):
        # Error in output
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)

        # Error in hidden layer
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            # Forward pass
            output = self.feedforward(X)

            # Backpropagation and weight update
            self.backpropagation(X, y, output)

            if (epoch + 1) % 1000 == 0:
                loss = np.mean(np.square(y - output))  # Mean Squared Error
                print(f"Epoch {epoch + 1}, Loss: {loss:.5f}")

    def predict(self, X):
        output = self.feedforward(X)
        return np.round(output)

# XOR Gate example
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0], [1], [1], [0]])

# Create a Neural Network with 2 input neurons, 2 hidden neurons, and 1 output neuron
nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1)

# Train the network
nn.train(X, y, epochs=10000)

# Predictions
print("\nPredictions after training:")
for i in range(len(X)):
    print(f"Input: {X[i]}, Predicted Output: {nn.predict(X[i].reshape(1, -1))}, Actual Output: {y[i]}")
