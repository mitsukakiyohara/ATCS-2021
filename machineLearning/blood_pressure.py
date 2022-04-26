"""
@author: Mitsuka Kiyohara
@version: 03/02/2022
@source: CodeHS
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

''' Load Data '''
data = pd.read_csv("data/blood_pressure.csv")

''' TODO: Create Linear Regression '''
# Get the values from x and y
x = data["Age"].values
y = data["Blood Pressure"].values

# Separate the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# Use reshape to turn the x values into 2D arrays:
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)

# Create the model
model = LinearRegression().fit(x_train,y_train)

# Find the slope and intercept. Each should be a float and rounded to two decimal places.
slope = round(float(model.coef_),2)
intercept = round(float(model.intercept_), 2)

# Print out the linear equation
print("Model equation: y =", slope, "x + ", intercept)

# Get the predicted y values for the x_test values - Return an array
predict = model.predict(x_test)

# Predict the the blood pressure of someone who is 43 years old.
# x_predict = 43
# prediction = model.predict([[x_predict]])

# Print out the prediction
#print("Prediction when x (age) is", x_predict, "is", prediction)

# Compare the actual and predicted values
print("Testing Linear Model with Testing Data:")
for index in range(len(x_test)):
    # Actual y value
    actual = y_test[index]
    # Predicted y value
    y_pred = round(predict[index], 2)
    # Test x value
    x_val = float(x_test[index])

    print("x value:", x_val, "predicted y val:", y_pred, "actual y val:", actual)


# Print out Pearson's correlation
# print("Pearson's Correlation: r = :", x_train.corr(predict))
r_squared = model.score(x_train, y_train)
print("R squared: ", r_squared)

''' Visualize Data '''
# set the size of the graph
plt.figure(figsize=(5, 4))

# label axes and create a scatterplot
plt.xlabel("Age")
plt.ylabel("Systolic Blood Pressure")
plt.title("Systolic Blood Pressure by Age")
plt.scatter(x_train,y_train,c="purple", label="Training Data")
plt.scatter(x_test, y_test, c="blue", label="Testing Data")
plt.scatter(x_test, predict, c="green", label="Predictions")

# print out best fit line
plt.plot(x_train, slope*x_train + intercept, c="red", label="Line of best fit")

plt.show()


