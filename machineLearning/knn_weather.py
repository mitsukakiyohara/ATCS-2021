import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score 

''' Visualize Data '''
data = pd.read_csv("data/weather.csv")

# Scatter plot with Seaborn 
g = sns.relplot(data=data, x="weather", y="temp", hue="outerwear")
plt.grid()
# plt.show()

''' Load Data '''
# Read features and classes 
feature_weather = data["weather"].values
feature_temp = data["temp"].values
classes = data["outerwear"].values

# Transform qualititative data to quantitative 
weather_transfomer = LabelEncoder().fit(feature_weather)
feature_weather = weather_transfomer.transform(feature_weather)

# Set the features to be a 2D array 
# Each day is its own array 
features = np.array([feature_weather, feature_temp]).transpose()

# Get the unique classes for labels 
class_labels = np.unique(data["outerwear"])

# Standardize data 
scaler = StandardScaler().fit(features)
features = scaler.transform(features)
# print(features)

# Split test and training data 
features_train, features_test, classes_train, classes_test = train_test_split(features, classes, test_size=0.2)

''' Create model '''
model = KNeighborsClassifier(n_neighbors=2).fit(features_train, classes_train)

''' Test model '''
classes_pred = model.predict(features_test)

# Accuracy 
print("Accuracy:", accuracy_score(classes_pred, classes_test))

# Confusion matrix 
cm = confusion_matrix(classes_test, classes_pred, labels=class_labels)

# Visualize Confusion Matrix 
cmd = ConfusionMatrixDisplay(cm, display_labels=class_labels)
cmd.plot()
plt.show()