import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load the data
df = pd.read_excel('./formattedData.xlsx')
# Print head of the dataset
print(df.head())

# Identify all column with non-numeric data
non_numeric_columns = ['Make', 'Model', 'Body_Type', 'Engine', 'Transmission', 'Drivetrain', 'Exterior_Colour', 'Interior_Colour', 'Fuel_Type']

# Show the distribution of the numerical columns
# df.hist(figsize=(12, 10))
# plt.show()

# Remove outliers from the numerical columns
numeric_columns = df.drop(non_numeric_columns, axis=1).columns
for column in numeric_columns:
    df = df[(df[column] >= df[column].quantile(0.01)) & (df[column] <= df[column].quantile(0.99))]

# Show new distribution of the numerical columns
print(df.describe())
# df.hist(figsize=(12, 10))
# plt.show()

# One-hot encode the non-numeric columns
df = pd.get_dummies(df, columns=non_numeric_columns)
# Print the head of the dataset
print(df.head())

# Split the data into train and test
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
# model = LinearRegression()
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Plot acutal vs predicted
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Price')
plt.show()