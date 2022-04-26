"""
Create a Multiple Linear Regression model to predict selling price of a house
"""
 # Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split

''' Organize data'''
data = pd.read_csv("data5/houses.csv")

# Organize our data into the correct format
x = data[["bathrooms", "lot_size_1000_sqft", "living_space_1000_sqft", "garages", "bedrooms", "age"]].values
y = data["selling_price"].values

# Split our training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

''' Create model '''
model = LinearRegression().fit(x_train, y_train)
r_squared = model.score(x_train, y_train)

# Print out the model information
print("Model Information:")
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
    x_bathrooms = x_test[index][0]
    x_lot_size_1000_sqft = x_test[index][1]
    x_living_space_1000_sqft = x_test[index][2]
    x_garages = x_test[index][3]
    x_bedrooms = x_test[index][4]
    x_age = x_test[index][5]

    print("x_bathrooms:", x_bathrooms, "x_lot_size_1000_sqft:", x_lot_size_1000_sqft, "x_living_space_1000_sqft:", x_living_space_1000_sqft, "x_garages:", x_garages, "x_bedrooms:", x_bedrooms, "x_age:", x_age, "predicted y val:", y_pred, "actual y val:", actual)

"""
x_bathrooms: 1.0 x_lot_size_1000_sqft: 4.455 x_living_space_1000_sqft: 0.988 x_garages: 1.0 x_bedrooms 3.0 x_age 56.0 predicted y val: 539140.29 actual y val: 389100
x_bathrooms: 1.0 x_lot_size_1000_sqft: 7.8 x_living_space_1000_sqft: 1.5 x_garages: 1.5 x_bedrooms 3.0 x_age 23.0 predicted y val: 614366.68 actual y val: 903840
"""
