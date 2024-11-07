import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def fit(self, X, y):
        # Initialize weights and bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                linear_output = np.dot(X[i], self.weights) + self.bias
                y_pred = self.activation_function(linear_output)
                # Update weights and bias based on prediction error
                error = y[i] - y_pred
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            linear_output = np.dot(X[i], self.weights) + self.bias
            y_pred.append(self.activation_function(linear_output))
        return np.array(y_pred)

# Define training data for OR gate
X_OR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_OR = np.array([0, 1, 1, 1])

# Define training data for AND gate
X_AND = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_AND = np.array([0, 0, 0, 1])

# Initialize Perceptron
perceptron_OR = Perceptron(learning_rate=0.1, epochs=10)
perceptron_AND = Perceptron(learning_rate=0.1, epochs=10)

# Train perceptron for OR gate
print("Training OR gate")
perceptron_OR.fit(X_OR, y_OR)

# Predictions for OR gate
predictions_OR = perceptron_OR.predict(X_OR)
print("Predictions for OR gate:")
for i, prediction in enumerate(predictions_OR):
    print(f"Input: {X_OR[i]}, Prediction: {prediction}, Actual: {y_OR[i]}")

# Train perceptron for AND gate
print("\nTraining AND gate")
perceptron_AND.fit(X_AND, y_AND)

# Predictions for AND gate
predictions_AND = perceptron_AND.predict(X_AND)
print("Predictions for AND gate:")
for i, prediction in enumerate(predictions_AND):
    print(f"Input: {X_AND[i]}, Prediction: {prediction}, Actual: {y_AND[i]}")
