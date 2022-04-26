"""
Create a Multiple Linear Regression model to predict the Cirrhosis death rate
"""
# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split

data = pd.read_csv("data5/drinking.csv")

''' Create multiple linear regression model '''

# Organize our data into the correct format 
x = data[["Urban_population_percentage", "Wine_consumption_per_capita", "Liquor_consumption_per_capita"]].values
y = data["Cirrhosis_death_rate"].values

# Split our training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Create model
model = LinearRegression().fit(x_train, y_train)
r_squared = model.score(x_train, y_train)

# Print out the model information
print("Model Information:")
# print("Wine_consumption_per_capita coefficient:", model.coef_[0])
# print("Liquor_consumption_per_capita coefficient:", model.coef_[1])
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
    x_wine = x_test[index][0]
    x_liquor = x_test[index][1]

    print("x_wine:", x_wine, "x_liquor:", x_liquor, "predicted y val:", y_pred, "actual y val:", actual)

"""
x_wine: 52 x_liquor: 7 predicted y val: 52.0 actual y val: 57.5
x_wine: 43 x_liquor: 4 predicted y val: 41.47 actual y val: 31.7
x_wine: 71 x_liquor: 11 predicted y val: 67.57 actual y val: 74.8
"""