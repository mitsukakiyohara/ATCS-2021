"""
Facial Regonition Program with Olivetti Dataset
Name: Mitsuka Kiyohara
Date: 4/21/22 
Block: B 

"""

''' Import Libraries '''
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt 

''' Load Data '''
data = np.load("olivetti_faces.npy")
target_labels = np.load("olivetti_faces_target.npy") 

''' Performing PCA '''
# Transform from matrix to vector form 
num_samples = data.shape[0]
image_size = data.shape[1] * data.shape[2] # 64 by 64
X = data.reshape((num_samples, image_size))

# print("X.shape = " + str(X.shape))

# Find optimum number of principle component (n_component)
pca = PCA()
pca.fit(X)
plt.figure(1, figsize=(12,8))
plt.plot(pca.explained_variance_, linewidth=2)

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

# USING PCA 
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

# USING PCA 
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

# USING PCA 
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

# USING PCA
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

# USING PCA
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
# Without PCA 
# Credits: Oanh Doan 
df = pd.DataFrame({'Method': model_names, 'Accuracy (%)': model_accuracies})
# df = df.sort_values(by=['Accuracy (%)'])
df = df.reset_index(drop=True)
df.head()

# With PCA 
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

'''Hyperparameter Tuning: GridSearchCV'''
# Credits: Satyam Kumar
# Create the Pipeline 
pipeline = Pipeline([('classifier', models[1])])
# Create a list of parameter dictionaries 
params = [param0, param1, param2, param3, param4]


# Initialize the hyperparameters for each dictionary 
# Comment out parameter values that I won't be using 
param0 = {}

# values for classifier__C: 10**-2, 10**-1, 10**0, 10**1, 10**2
param0['classifier__C'] = [10**-2, 10**-1, 10**0, 10**1, 10**2]

param0['classifier__penalty'] = ['l1', 'l2']

# values for classifier__class_weight: None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}
param0['classifier__class_weight'] = [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}]
param0['classifier'] = [models[0]]

param1 = {}
param1['classifier__max_depth'] = [5, 10, 20]

# values for classifier__class_weight: None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}
param1['classifier__class_weight'] = [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}]
param1['classifier'] = [models[1]]

param2 = {}

# values for classifier__n_neighbors: 2,5,10,25,50
param2['classifier__n_neighbors'] = [2,5,10,25,50]
param2['classifier'] = [models[2]]

param3 = {}
# values for classifier__C: 10**-2, 10**-1, 10**0, 10**1, 10**2
param3['classifier__C'] = [10**-2, 10**-1, 10**0, 10**1, 10**2]
param3['classifier__class_weight'] = [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}]
param3['classifier'] = [models[3]]

param4 = {}
param4['classifier__alpha'] = [10**0, 10**1, 10**2]
param4['classifier'] = [models[4]]

# Train the grid search model by searchinbg every parameter combination within each dictionary
gs = GridSearchCV(pipeline, params, cv=3, n_jobs=-1, scoring='roc_auc').fit(X_train, y_train)

# Best performing model and its corresponding hyperparameters
gs.best_params_

# ROC-AUC score for the best model
gs.best_score_

# Test data performance
print("Precision:",precision_score(rs.predict(X_test), y_test))
print("Recall:",recall_score(rs.predict(X_test), y_test))
print("ROC AUC Score:",roc_auc_score(rs.predict(X_test), y_test))
