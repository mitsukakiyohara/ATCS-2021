"""
Creates a logistic regression model to determine whether a diamond is premium or fair 
"""
# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix 

''' Load Data '''
data = pd.read_csv("data5/diamonds.csv")

# Replace qualitative data with binary (quantitative)
data["cut"].replace(["Fair", "Premium"], [0, 1], inplace=True) # Fair = 0, Premium = 1

x = data[["depth", "carat", "table"]].values 
y = data["cut"].values 

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
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# Calculate accuracy 
accuracy = model.score(x_test, y_test)
print("Accuracy:", accuracy)

''' Make a new prediction '''
depth = float(input("What is the depth?\n"))
carat = float(input("What is the carat?\n"))
table = float(input("What is the table?\n"))

# Scale the inputs 
x_pred = [[depth, carat, table]]
x_pred = scaler.transform(x_pred)

# Make and print prediction (0 or 1, fair or premium)
if model.predict(x_pred)[0] == 1:
    print("This diamond cut is most likely premium")
else: 
    print("This diamond cut is most likely fair") 

"""
Test data: 
depth = 65.8, carat = 1, table = 60 --> fair
depth = 65.8, carat = 0.5, table = 60 --> fair
depth = 65.8, carat = 1, table = 50 --> premium
depth = 59.0, carat = 0.6, table = 40 --> premium
"""
