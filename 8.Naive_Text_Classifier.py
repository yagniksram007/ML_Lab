# Implement the naive bayes classifier for a sample trainng dataset stored in a csv file

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load the Dataset
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model
model = GaussianNB()
model.fit(X_train, y_train)

# Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Output:
# Accuracy: 0.9
# Classification Report:
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00        10
#            1       0.88      0.78      0.82         9
#            2       0.83      0.91      0.87        11

#     accuracy                           0.90        30
#    macro avg       0.90      0.90      0.90        30
# weighted avg       0.90      0.90      0.90        30


