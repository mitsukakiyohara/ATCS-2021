"""
Create a Multiple Linear Regression model to predict how many murders happen per year
"""
# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split

''' Organize data'''
data = pd.read_csv("data5/homicides.csv")

# Organize our data into the correct format
x = data[["Inhabitants", "Percent_with_income_below_5000", "Percent_unemployed"]].values
y = data["Murders_per_year_per_million"].values

# Split our training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

''' Create model '''
model = LinearRegression().fit(x_train, y_train)
r_squared = model.score(x_train, y_train)

# Print out the model information
print("Model Information:")
# print("Inhabitants coefficient:", model.coef_[0])
# print("Percent_with_income_below_5000 coefficient:", model.coef_[1])
# print("Percent_unemployed coefficient:", model.coef_[2])
# print("Intercept:", round(float(model.intercept_), 2))
print("R squared values:", r_squared)
print()

# Get the predicted y values for the x_test values
predict = model.predict(x_test)

# Compare the actual and predicted values
print("Testing Linear Model with Testing Data:")
for index in range(len(x_test)):
    # Actual y value
    actual = y_test[index]
    # Predicted y value
    y_pred = round(predict[index], 2)

    # Test x value
    x_inhabitants = x_test[index][0]
    x_incomebelow5000 = x_test[index][1]
    x_unemployed = x_test[index][2]

    print("x_inhabitants:", x_inhabitants, "x_incomebelow5000:", x_incomebelow5000, "x_unemployed:", x_unemployed, "predicted y val:", y_pred, "actual y val:", actual)

"""
x_inhabitants: 1964000.0 x_incomebelow5000: 20.2 x_unemployed: 6.4 predicted y val: 19.09 actual y val: 20.9
x_inhabitants: 749000.0 x_incomebelow5000: 14.3 x_unemployed: 6.4 predicted y val: 11.73 actual y val: 9.6
"""