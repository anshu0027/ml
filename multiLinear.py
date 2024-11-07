import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

print("IU2141230160 - Anshu Patel")

data = fetch_california_housing(as_frame=True)
df = data.frame

print(df.head())

df.fillna(df.median(), inplace=True)

X = df[['MedInc', 'HouseAge', 'AveRooms']]
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

area = 3000
bedrooms = 4
age = 15

new_data = pd.DataFrame([[area, age, bedrooms]], columns=['MedInc', 'HouseAge', 'AveRooms'])  # Adjust column names
predicted_price = model.predict(new_data)

print(f'Predicted Price: {predicted_price[0]}')
