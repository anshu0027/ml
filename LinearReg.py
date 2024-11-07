import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

print("IU2141230160 - Anshu Patel")

data = fetch_california_housing(as_frame=True)
df = data.frame

# Selecting a subset of 200 rows for simplicity
df = df.sample(n=200, random_state=42)

X = df[['MedInc']]  # Independent variable: Median Income
y = df['MedHouseVal']  # Dependent variable: Median House Value

# Print basic dataset information
print(f"Total observations in data: {df.shape[0]}")
print(f"Number of independent variables: {X.shape[1]}")
print(f"Dependent variable: {y.name}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Calculate RMSE, SSE, and R² Score
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
sse = np.sum((y_test - y_pred) ** 2)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f'RMSE: {rmse:.2f}')
print(f'SSE: {sse:.2f}')
print(f'R² Score: {r2:.2f}')

# Visualize the regression line
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.title('Simple Linear Regression')
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.legend()
plt.show()
