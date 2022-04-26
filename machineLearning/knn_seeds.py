from optparse import Values
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score 


data = pd.read_csv("data/seeds.csv")

''' Load Data '''
# Read features and classes 
feature_area = data["area"].values
feature_perimeter = data["perimeter"].values
feature_compactness = data["compactness"].values
feature_kernel_length = data["kernel_length"].values
feature_kernel_width = data["kernel_width"].values
feature_asymmetry_coef = data["asymmetry_coef"].values
feature_groove_length = data["groove_length"].values
classes = data["seed"].values

# Set features to be a 2D arrray 
features = np.array([feature_area, feature_perimeter, feature_compactness, feature_kernel_length, feature_kernel_width, feature_asymmetry_coef, feature_groove_length]).transpose()

# Get unique classes for lables
class_labels = np.unique(data["seed"])

# Standardize data 
scaler = StandardScaler().fit(features)
features = scaler.transform(features)

# Split test and training data 
features_train, features_test, classes_train, classes_test = train_test_split(features, classes, test_size=0.2)


''' Create/Test model with Hypertuning '''
k_range = range(2, 150)
# no_neighbors = np.arange(1, 9)
test_accuracy = []

for k in k_range: 
    # Create model 
    model = KNeighborsClassifier(n_neighbors=k).fit(features_train, classes_train)
    # Test model 
    classes_pred = model.predict(features_test)
    # Store accuracy 
    test_accuracy.append(accuracy_score(classes_pred, classes_test))


''' Graph Accuracy of Predictions '''
plt.plot(k_range, test_accuracy)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()

# Confusion matrix 
# cm = confusion_matrix(classes_test, classes_pred, labels=class_labels)

# Visualize Confusion Matrix 
# cmd = ConfusionMatrixDisplay(cm, display_labels=class_labels)
# cmd.plot()
# plt.show()