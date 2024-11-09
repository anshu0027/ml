# Pandas
import pandas as pd
print("IU2141230160")
data = {
'Name': ['AP', 'Kismat', 'Ansh'], 
'Age': [21, 22, 30],
'City': ['New York', 'San Francisco', 'Los Angeles']
}
df = pd.DataFrame(data)
print("DataFrame:")
print(df)
ages = df['Age']
print("\nAges:")
print(ages)
filtered_df = df[df['Age'] > 28]
print("\nFiltered DataFrame (Age > 28):")
print(filtered_df)

# matplotlib
import matplotlib.pyplot as plt
x = [1, 3, 5, 7, 9]
y = [2, 4, 5, 7, 11]
plt.plot(x, y, marker='o')
plt.title("Sample Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.show()
