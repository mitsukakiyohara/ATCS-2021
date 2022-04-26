"""
A Machine Learning algorithm to predict car prices

@author: Mitsuka Kiyohara
@version: 02/23/2022
@source: CodeHS
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split

''' Load Data '''
data = pd.read_csv("data/car.csv")
x_1 = data["miles"]
x_2 = data["age"]
y = data["Price"]

''' Visualize Data '''
fig, graph = plt.subplots(2)
graph[0].scatter(x_1, y)
graph[0].set_xlabel("Total Miles")
graph[0].set_ylabel("Price")

graph[1].scatter(x_2, y)
graph[1].set_ylabel("Price")
graph[1].set_xlabel("Car Age")

print("Correlation between Total Miles and Car Price:", x_1.corr(y))
print("Correlation between Age and Car Price:", x_2.corr(y))

plt.tight_layout()
# plt.show()

''' TODO: Create Linear Regression '''
# Reload and/or reformat the data to get the values from x and y
x = data[["miles", "age"]].values
y = data["Price"].values

# Separate data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Create multivariable linear regression model
model = LinearRegression().fit(x_train, y_train)
r_squared = model.score(x_train, y_train)

# Find and print the coefficients, intercept, and r squared values. Each rounded to two decimal places.
print("Model Information:")
print("Mileage coefficient:", round(float(model.coef_[0]), 2))
print("Age coefficient:", round(float(model.coef_[1]), 2))
print("Intercept:", round(float(model.intercept_), 2))
print("R squared values:", r_squared)
print()


print(model.predict([[150000, 20]]))

# Test the model
predict = model.predict(x_test)

# Compare the actual and predicted values. Print out the actual vs the predicted values
print("Testing Linear Model with Testing Data:")
for index in range(len(x_test)):
    # Actual y value
    actual = y_test[index]
    # Predicted y value
    y_pred = round(predict[index], 2)

    # Test x value
    x_mileage = x_test[index][0]
    x_age = x_test[index][1]

    print("x_mileage:", x_mileage, "x_age:", x_age, "predicted y val:", y_pred, "actual y val:", actual)
