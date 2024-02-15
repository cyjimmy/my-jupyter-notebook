import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data
df = pd.read_excel('./formattedData.xlsx')

# Print info of the data
print(df.head())
print(df.columns.to_list())

# Find all column with non-numeric data
non_numeric_columns = []
for column in df.columns:
    if df[column].dtype == 'object':
        non_numeric_columns.append(column)

# Print the non-numeric columns
print(non_numeric_columns)


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

