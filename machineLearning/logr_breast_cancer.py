"""
Logistic regression model to determine if someone will survive breast cancer 
"""
# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix 

''' Load Data '''
data = pd.read_csv("data/breast_cancer.csv")

x = data[["Age", "Nodes"]].values 
# x = data[["Age", "Year", "Nodes"]].values 
y = data["Survived_5_Years"].values 

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
age = int(input("How old is the patient?\n"))
nodes = int(input("How many nodes did the patient have?\n"))
year = int(input("In what year did you have surgery?\n"))

# Scale the inputs 

x_pred = [[age, nodes]]
# x_pred = [[age, year, nodes]]
x_pred = scaler.transform(x_pred)

# Make and print prediction (0 or 1, die within 5 years or survive)
if model.predict(x_pred)[0] == 1:
    print("This customer will likely die from breast cancer within 5 years.")
else: 
    print("This customer will likely not die from breast cancer within 5 years.") 