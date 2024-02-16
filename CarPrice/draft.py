import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load the data
df = pd.read_excel('./formattedData.xlsx')

# Identify all column numeric and non-numeric data
non_numeric_columns = ['Make', 'Model', 'Body_Type', 'Engine', 'Transmission', 'Drivetrain', 'Exterior_Colour', 'Interior_Colour', 'Fuel_Type']
numeric_columns = df.drop(non_numeric_columns, axis=1).columns

# One-hot encode the non-numeric columns
df = pd.get_dummies(df, columns=non_numeric_columns)

# Split training and testing data
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Combine the training data
train_data = X_train.copy()
train_data['Price'] = y_train

# # Remove outliers from the numerical columns
# for column in numeric_columns:
#     train_data = train_data[(train_data[column] >= train_data[column].quantile(0.01)) & (train_data[column] <= train_data[column].quantile(0.99))]

# Split the training data
X_train = train_data.drop('Price', axis=1)
y_train = train_data['Price']

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_error(y_test, y_pred) / y_test.mean()
print(f'Mean Absolute Error: {mae}')
print(f'Mean Absolute Percentage Error: {mape}')