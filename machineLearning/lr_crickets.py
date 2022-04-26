import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split

''' Load Data '''
data = pd.read_csv("data/chirping.csv")

# Independent 
x = data["Temp"].values
# Dependent 
y = data["Chirps"].values

# Separate the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# Turn x into a 2D array 
x_train = x_train.reshape(-1, 1)

''' Create the model '''
model = LinearRegression().fit(x_train,y_train)
slope = round(float(model.coef_),2)
intercept = round(float(model.intercept_), 2)
r_squared = model.score(x_train, y_train)

''' Test the model '''
# x_predict = 77
# prediction = model.predict([[x_predict]])

# Reshape x_test to a 2D array
x_test = x_test.reshape(-1, 1)

# Get the predicted y values for the x_test values - Return an array
predict = model.predict(x_test)

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

''' Print results '''
print("Model equation: y =", slope, "x + ", intercept)
print("R squared: ", r_squared)
# print("Prediction when x is", x_predict, y_pred)

''' Visualization '''
plt.figure(figsize=(5,4))

#Scatterplot
plt.scatter(x_train,y_train,c="purple", label="Training Data")
plt.scatter(x_test, y_test, c="blue", label="Testing Data")
plt.scatter(x_test, predict, c="green", label="Predictions")
plt.xlabel("Temperature (F)")
plt.ylabel("Chirps per Minute")
plt.title("Cricket Chirps by Temperature")

plt.plot(x_train, slope*x_train + intercept, c="red", label="Line of best fit")
plt.legend()
plt.show()