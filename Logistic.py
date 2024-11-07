import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print("Total observations in data:", df.shape[0])

X, y = df.drop(columns=['target']), df['target']

print("Number of independent variables:", X.shape[1])
print("Dependent variable:", y.name)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
logreg = LogisticRegression(max_iter=10000)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))

# Suggestions for improvement
print("Steps taken for improvement:")
print("1. Feature Engineering: Create additional relevant features.")
print("2. Hyperparameter Tuning: Use Grid Search or Random Search for optimal hyperparameters.")
print("3. Model Selection: Compare with algorithms like SVM, Random Forest, or Gradient Boosting.")
print("4. Class Imbalance: Use oversampling, undersampling, or cost-sensitive learning.")
print("5. Cross-Validation: Implement k-fold cross-validation for better generalization.")

# Prediction function
def predict_cancer(features):
    features_df = pd.DataFrame([features], columns=X.columns)  # Create DataFrame with correct column names
    return "Malignant" if logreg.predict(features_df)[0] == 1 else "Benign"

# Example prediction
features = X_test.iloc[0]
print("Prediction for the first test sample features:", predict_cancer(features))
