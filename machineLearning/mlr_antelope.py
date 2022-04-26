"""
Create a Multiple Linear Regression model to predict Antelope population
"""
# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split

'''
Visualizing existing data
'''

data = pd.read_csv("data/antelope.csv")
x_1 = data["Annual Precipitation"]
x_2 = data["Winter Severity"]
y = data["Fawn"]

fig, graph = plt.subplots(2)

graph[0].scatter(x_1, y)
graph[0].set_xlabel("Annual Precipitation")
graph[0].set_ylabel("Fawn")

graph[1].scatter(x_2, y)
graph[1].set_xlabel("Winter Severity")
graph[1].set_ylabel("Fawn")

plt.tight_layout()
# plt.show()

print("Corr between annual precipitation and fawn population:", x_1.corr(y))
print("Corr between winter severity and fawn population:", x_2.corr(y))

''' Create multiple linear regression model '''

# Organize our data into the correct format (REMEMBER .VALUES)
x = data[["Annual Precipitation", "Winter Severity"]].values
y = data["Fawn"].values

# Split our training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Create model
model = LinearRegression().fit(x_train, y_train)

# Print out the model information
print("Model Information:")
print("Annual Precipitation coefficient:", model.coef_[0])
print("Winter Severity coefficient:", model.coef_[1])
print("Intercept:", round(float(model.intercept_), 2))
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
    x_precipitation = x_test[index][0]
    x_winter = x_test[index][1]

    print("x_precipitation:", x_precipitation, "x_winter:", x_winter, "predicted y val:", y_pred, "actual y val:", actual)