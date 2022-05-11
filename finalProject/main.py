"""
Facial Regonition Program with Olivetti Dataset (with Cross Validation and Hyperparameter tuning) 
Name: Mitsuka Kiyohara
Date: 5/10/22 
Block: B 

"""

''' Import Libraries '''
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.pipeline import Pipeline


''' Load Data '''
data = np.load("olivetti_faces.npy")
target_labels = np.load("olivetti_faces_target.npy") 

''' Performing PCA '''
# Transform from matrix to vector form 
num_samples = data.shape[0]
image_size = data.shape[1] * data.shape[2] # 64 by 64
X = data.reshape((num_samples, image_size))

# Find optimum number of principle component (n_component)
pca = PCA()
pca.fit(X)
plt.figure(1, figsize=(12,8))
plt.plot(pca.explained_variance_, linewidth=2)

# Decide most optimal PCA component size from graph
plt.xlabel('n_components')
plt.ylabel('Explained Variances')
plt.title('Variance vs. Components') 
plt.show()

''' Split data and target into random test and train subsets '''
# We will be splitting it 9:1 (90% for training, 10% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.1, stratify=target, random_state=0)

n_components = 70 
pca = PCA(n_components=n_components, whiten=True).fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

''' Plot the average face from all of the samples '''
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
axs.imshow(pca.mean_.reshape((64, 64)), cmap="gray")
axs.set_xticks([])
axs.set_yticks([])
axs.set_title('Average Face')

''' Testing Different ML Models '''
# Store accuracies on the machine learning methods for comparison at the end 
model_names = []
model_accuracies = []

model_names_pca = []
model_accuracies_pca = []

''' Logistic Regression '''
# WITHOUT PCA
lr = LogisticRegression().fit(X_train, y_train)

lr_accuracy = round(lr.score(X_test, y_test) * 100, 2)

print("lr_accuracy is %", lr_accuracy)

model_names.append("Logistic Regression")
model_accuracies.append(lr_accuracy)
y_pred = lr.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# WITH PCA 
lr = LogisticRegression().fit(X_train_pca, y_train)

lr_accuracy_pca = round(lr.score(X_test_pca, y_test) * 100, 2)

print("lr_accuracy is %", lr_accuracy_pca)

model_names_pca.append("Logistic Regression")
model_accuracies_pca.append(lr_accuracy_pca)
y_pred = lr.predict(X_test_pca)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

''' Random Forest Classifier '''
# WITHOUT PCA
rf = RandomForestClassifier(n_estimators=400, random_state=1).fit(X_train, y_train)
# Note that n_estimators is chosen at random. While it is generally assumed that larger
# the number of trees, the better, the optimal number of estimators or trees will be found
# via GridSearch

rf_accuracy = round(rf.score(X_test, y_test) * 100, 2)

print("rf_accuracy is %", rf_accuracy)

model_names.append("Random Forest")
model_accuracies.append(rf_accuracy)
y_pred = rf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# WITH PCA 
rf = RandomForestClassifier(n_estimators=400, random_state=1).fit(X_train_pca, y_train)
# Note that n_estimators is chosen at random. While it is generally assumed that larger
# the number of trees, the better, the optimal number of estimators or trees will be found
# via GridSearch

rf_accuracy_pca = round(rf.score(X_test_pca, y_test) * 100, 2)

print("rf_accuracy is %", rf_accuracy_pca)

model_names_pca.append("Random Forest")
model_accuracies_pca.append(rf_accuracy_pca)
y_pred = rf.predict(X_test_pca)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

''' KNN (K-Nearest Neighbors) '''
# WITHOUT PCA
knn = KNeighborsClassifier(n_neighbors=2).fit(X_train, y_train)

knn_accuracy = round(knn.score(X_test, y_test) * 100, 2)

print("knn_accuracy is %", knn_accuracy)

model_names.append("KNN")
model_accuracies.append(knn_accuracy)
y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# WITH PCA 
knn = KNeighborsClassifier(n_neighbors=2).fit(X_train_pca, y_train)

knn_accuracy_pca = round(knn.score(X_test_pca, y_test) * 100, 2)

print("knn_accuracy is %", knn_accuracy_pca)

model_names_pca.append("KNN")
model_accuracies_pca.append(knn_accuracy_pca)
y_pred = knn.predict(X_test_pca)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

''' SVM (Support Vector Machines) '''
# WITHOUT PCA
svm = SVC(kernel='linear', random_state=0).fit(X_train, y_train)

svm_accuracy = round(svm.score(X_test, y_test) * 100, 2)

print("svm_accuracy is %", svm_accuracy)

model_names.append("SVM")
model_accuracies.append(svm_accuracy)
y_pred = svm.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# WITH PCA
svm = SVC(kernel='linear', random_state=0).fit(X_train_pca, y_train)

svm_accuracy_pca = round(svm.score(X_test_pca, y_test) * 100, 2)

print("svm_accuracy is %", svm_accuracy_pca)

model_names_pca.append("SVM")
model_accuracies_pca.append(svm_accuracy_pca)
y_pred = svm.predict(X_test_pca)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

''' Naive Bayes '''
# WITHOUT PCA
nb = GaussianNB().fit(X_train, y_train)

nb_accuracy = round(nb.score(X_test, y_test) * 100, 2)

print("nb_accuracy is %", nb_accuracy)

model_names.append("Naive Bayes")
model_accuracies.append(nb_accuracy)
y_pred = nb.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# WITH PCA
nb = GaussianNB().fit(X_train_pca, y_train)

nb_accuracy_pca = round(nb.score(X_test_pca, y_test) * 100, 2)

print("nb_accuracy is %", nb_accuracy_pca)

model_names_pca.append("Naive Bayes")
model_accuracies_pca.append(nb_accuracy_pca)
y_pred = nb.predict(X_test_pca)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

''' Comparing Accuracies of Different ML Models '''
# WITHOUT PCA 
# Credits: Oanh Doan 
df = pd.DataFrame({'Method': model_names, 'Accuracy (%)': model_accuracies})
# df = df.sort_values(by=['Accuracy (%)'])
df = df.reset_index(drop=True)
df.head()

# WITH PCA 
df_pca = pd.DataFrame({'Method': model_names_pca, 'Accuracy (%)': model_accuracies_pca})
# df = df.sort_values(by=['Accuracy (%)'])
df_pca = df_pca.reset_index(drop=True)
df_pca.head()

''' Cross Validation: K-Fold '''
models=[]
models.append(LogisticRegression())
models.append(RandomForestClassifier(n_estimators=400))
models.append(KNeighborsClassifier(n_neighbors=2))
models.append(SVC())
models.append(GaussianNB())

# Convergence warning for logistic regression - increase size of dataset, increase num of iterations, or scale data
# WITHOUT PCA
model_cv_scores = []
for model in models: 
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kfold)
    
    # print("{} mean cross validation scores: :{:.2f}".format(name, cv_scores.mean()))
    model_cv_scores.append(round(cv_scores.mean() * 100, 2))

df['CV Score (%)'] = model_cv_scores
# df

# WITH PCA
model_cv_scores_pca = []
for model in models: 
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    cv_scores_pca = cross_val_score(model, X_train_pca, y_train, cv=kfold)
    
    # print("{} mean cross validation scores: :{:.2f}".format(name, cv_scores.mean()))
    model_cv_scores_pca.append(round(cv_scores_pca.mean() * 100, 2))

df_pca['CV Score (%)'] = model_cv_scores_pca
# df_pca

'''Hyperparameter Tuning: GridSearchCV and RandomizedSearchCV'''
# Credits: Satyam Kumar

# Initialize the hyperparameters for each dictionary 
# Note: Had to cut down number of values being tested due to processing time. 
# Commented the full values I would have tested 

param0 = {}

# values for C: 10**-2, 10**-1, 10**0, 10**1, 10**2
param0['C'] = [10**-2, 10**0, 10**2]
param0['penalty'] = ["none", 'l2']

# values for class_weight: None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}
param0['classifier__class_weight'] = [None, {0:1,1:5}, {0:1,1:10}]
param0['classifier'] = [models[0]]

param1 = {}
param1['max_depth'] = [5, 10, 20]

# values for class_weight: None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}
param1['class_weight'] = [None, {0:1,1:5}, {0:1,1:10}]
param1['classifier'] = [models[1]]

param2 = {}

# values for n_neighbors: 2,5,10,25,50
param2['n_neighbors'] = [5,10,50]
param2['classifier'] = [models[2]]

param3 = {}
# values for classifier__C: 10**-2, 10**-1, 10**0, 10**1, 10**2
param3['C'] = [10**-2, 10**0, 10**2]

# values for classifier__class_weight: [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}]
param3['class_weight'] = [None, {0:1,1:5}, {0:1,1:10}]
param3['classifier'] = [models[3]]

param4 = {}
param4['alpha'] = [10**0, 10**1, 10**2]
param4['classifier'] = [models[4]]

# Create the Pipeline 
pipeline = Pipeline([('classifier', models[1])])

# Create a list of parameter dictionaries 
params = [param0, param1, param2, param3, param4]

# Train the grid search model by searching every parameter combination within each dictionary (PIPELINE SOLUTION)
# gs = GridSearchCV(pipeline, params, cv=3, n_jobs=-1, scoring='roc_auc').fit(X_train, y_train)

# WITH PCA 
# gs = GridSearchCV(pipeline, params, cv=3, n_jobs=-1, scoring='roc_auc').fit(X_train_pca, y_train)


# Applying GridSearchCV on the Logistic Regression model (JUST ONE MODEL) 
gs = GridSearchCV(models[0], param0, cv=2, n_jobs=-1, scoring='accuracy').fit(X_train, y_train)                                                      
# Applying RandomizedSearch CV on the Logistic Regression model (JUST ONE MODEL) 
rs = RandomizedSearchCV(models[0], param0, n_iter=100, n_jobs=-1, cv=2, scoring='accuracy').fit(X_train, y_train)   

# Best performing model and its corresponding hyperparameters from GridSearchCV (for RandomizedSearchCV, replace gs with rs)
gs.best_params_

# Mean cross validated score for the best model (for RandomizedSearchCV, replace gs with rs)
gs.best_score_

# Accuracy using newly found hyperparameters 
# print(accuracy_score(rgs.predict(y_test, y_pred)))