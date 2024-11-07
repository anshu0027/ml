# Import the necessary libraries
import numpy as np  # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
                            n_clusters_per_class=1, random_state=42)

# Create a DataFrame
df = pd.DataFrame(X, columns=['Age', 'EstimatedSalary'])
df['Purchased'] = y

# Split the dataset into features (X) and target (y)
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test data
predictions = rf_classifier.predict(X_test)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

# Print the results
print(f'Accuracy: {accuracy:.2f}')
print(f'\nClassification Report:')
print(report)

# Scatter plot
plt.scatter(X['Age'], X['EstimatedSalary'], c=y, cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.title('Scatterplot of Age vs Estimated Salary')
plt.show()

# Histograms
plt.figure(figsize=(12, 4))

# Histogram of Age
plt.subplot(1, 2, 1)
plt.hist(X['Age'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Histogram of Age')

# Histogram of Estimated Salary
plt.subplot(1, 2, 2)
plt.hist(X['EstimatedSalary'], bins=20, color='salmon', edgecolor='black')
plt.xlabel('Estimated Salary')
plt.ylabel('Frequency')
plt.title('Histogram of Estimated Salary')

plt.tight_layout()
plt.show()
