# Implement random forest classifier using python programming

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report

# Load the iris dataset 
iris = load_iris()

X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a Random Forest Classifier with 100 trees
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Predict on the test data
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)

print("Accuracy:", accuracy)
print('Classification Report:')
print(report)

# Output:
# Accuracy: 1.0
# Classification Report:
#               precision    recall  f1-score   support

#       setosa       1.00      1.00      1.00        19
#   versicolor       1.00      1.00      1.00        13
#    virginica       1.00      1.00      1.00        13

#     accuracy                           1.00        45
#    macro avg       1.00      1.00      1.00        45
# weighted avg       1.00      1.00      1.00        45