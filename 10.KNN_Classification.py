# Implement KNN classifier algorithm with an appropriate dataset and analyse the results

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the kNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn.predict(X_test)

# Analyze the results
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy:{accuracy:.2f}\n')

conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix,"\n")

class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)

# Output:
# Accuracy:0.98

# Confusion Matrix:
# [[19  0  0]
#  [ 0 17  0]
#  [ 0  1 16]]

# Classification Report:
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00        19
#            1       0.94      1.00      0.97        17
#            2       1.00      0.94      0.97        17

#     accuracy                           0.98        53
#    macro avg       0.98      0.98      0.98        53
# weighted avg       0.98      0.98      0.98        53

