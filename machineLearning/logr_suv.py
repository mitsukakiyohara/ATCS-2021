"""
Creates a logistic regression model to determine if someone will buy an SUV
"""
# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix 

''' Load Data '''
data = pd.read_csv("data/suv.csv")

# Replace qualitative data with binary (quantitative)
data["Gender"].replace(["Male", "Female"], [0, 1], inplace=True) # Male = 0, Female = 1

x = data[["Age", "EstimatedSalary", "Gender"]].values # 2D array, each array = 1 person 
y = data["Purchased"].values 

# Standardize data
scaler = StandardScaler().fit(x)
x = scaler.transform(x)

# Split our training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

''' Create Model '''
model = LogisticRegression().fit(x_train, y_train)

# Get weight for logistic regression equation 
coef = model.coef_[0]
print("Weights for model:")
print(coef)
print()

''' Test Model '''
y_pred = model.predict(x_test)

# Get confusion matrix 
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate accuracy 
accuracy = model.score(x_test, y_test)
print("Accuracy:", accuracy)

''' Make a new prediction '''
age = int(input("How old is the customer?\n"))
gender = int(input("Is the customer male (0) or female (1)?\n"))
salary = int(input("How much does the customer make in a year?\n"))

# Scale the inputs 
x_pred = [[age, salary, gender]]
x_pred = scaler.transform(x_pred)

# Make and print prediction (0 or 1, buy or not buy)
if model.predict(x_pred)[0] == 1:
    print("This customer will likely buy an SUV")
else: 
    print("This customer will not likely buy an SUV") 



