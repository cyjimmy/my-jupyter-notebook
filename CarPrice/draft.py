import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the data
df = pd.read_excel('./formattedData.xlsx')
# Print head of the dataset
print(df.head())

# Plot histogram of numerical columns
df.hist(figsize=(13, 10))
plt.show()


# # Plot price distribution
# plt.hist(df['Price'])
# plt.title('Price')
# plt.show()

# # Remove upper 1% of the price
# q = df['Price'].quantile(0.99)
# df = df[df['Price'] < q]

# # Plot price distribution
# plt.hist(df['Price'])
# plt.title('Price')
# plt.show()

# # Plot all columns distribution
# df = df.astype(str)
# for column in df.columns:
#     print(column)
#     plt.hist(df[column])
#     plt.title(column)
#     plt.show()

# # Find all column with non-numeric data
# non_numeric_columns = []
# for column in df.columns:
#     if df[column].dtype == 'object':
#         non_numeric_columns.append(column)
# # Print the non-numeric columns
# print(non_numeric_columns)

# # One-hot encode the non-numeric columns
# df = pd.get_dummies(df, columns=non_numeric_columns)
# # Print the head of the dataset
# print(df.head())

# # Split the data into train and test
# X = df.drop('Price', axis=1)
# y = df['Price']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train the model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Evaluate the model
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')
# mae = mean_absolute_error(y_test, y_pred)
# print(f'Mean Absolute Error: {mae}')