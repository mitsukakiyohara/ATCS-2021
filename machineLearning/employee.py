"""
Creates a logistic regression model to determine someone's job attrition
"""
# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix 

''' Load Data '''
data = pd.read_csv("data5/employees.csv")

# Replace qualitative data with binary (quantitative)
data["Gender"].replace(["Male", "Female"], [0, 1], inplace=True) # Male = 0, Female = 1
data["MaritalStatus"].replace(["Single", "Divorced", "Married"], [0, 1, 2], inplace=True) # Single = 0, Divorced = 1, Married = 2
data["Attrition"].replace(["No", "Yes"], [0, 1], inplace=True) # No = 0, Yes = 1 

x = data[["Age", "DistanceFromHome", "MaritalStatus", "MonthlyIncome"]].values
y = data["Attrition"].values

# Standardize data
scaler = StandardScaler().fit(x)
x = scaler.transform(x)

# Split our training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

''' Create Model '''
model = LogisticRegression().fit(x_train, y_train)

''' Test Model '''
y_pred = model.predict(x_test)

# Get confusion matrix 
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate accuracy 
accuracy = model.score(x_test, y_test)
print("Accuracy:", accuracy)

''' Make a new prediction '''
age = int(input("How old are you?\n"))
distance = int(input("How far is your home from workplace?\n"))
marriage = int(input("What is your marital status? (0 for single, 1 for divorced, 2 for married)\n"))
income = int(input("How much do you make in a month?\n"))

# Scale the inputs 
x_pred = [[age, distance, marriage, income]]
x_pred = scaler.transform(x_pred)

# Make and print prediction (0 or 1, )
if model.predict(x_pred)[0] == 1:
    print("This person will likely leave their job.")
else: 
    print("This person will not likely leave their job.") 

"""
age: 18, distance: 2, income: 6573 --> no
age: 43, distance: 12, income: 10000 --> no
"""
