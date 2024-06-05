# Implement and demonstrate the working of the decision tree algorithm

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Sample data
employee_data = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5, 6],
    'Name': ['John Doe', 'Jane Smith', 'Bob Brown', 'Unknown', 'Alice Johnson', 'Michael White'],
    'Department': ['Engineering', 'HR', 'Sales', 'Marketing', 'Engineering', 'HR'],
    'Salary': [60000, 59000, 52000, 55000, 70000, 58000],
    'JoiningDate': ['2020-01-15', '2019-07-10', '2018-03-22', '2021-11-30', '2019-05-24', '2017-12-12']
})

# Data Cleaning
employee_data['Name'].fillna('Unknown', inplace=True)
average_salary = employee_data['Salary'].mean()
employee_data['Salary'].fillna(average_salary, inplace=True)

# Calculate YearsOfService
employee_data['JoiningDate'] = pd.to_datetime(employee_data['JoiningDate'])
current_date = pd.to_datetime('2024-06-01')
employee_data['YearsOfService'] = (current_date - employee_data['JoiningDate']).dt.days / 365.25

# Create a new feature for salary classification
employee_data['AboveAverageSalary'] = employee_data['Salary'] > average_salary

# Encode categorical variables
label_encoder = LabelEncoder()
employee_data['Department'] = label_encoder.fit_transform(employee_data['Department'])

# Features and target
X = employee_data[['Department', 'YearsOfService']]
y = employee_data['AboveAverageSalary']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Decision Tree model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict on the test data
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Output:
# Accuracy: 1.0
# Classification Report:
#               precision    recall  f1-score   support

#        False       1.00      1.00      1.00         1
#         True       1.00      1.00      1.00         1

#     accuracy                           1.00         2
#    macro avg       1.00      1.00      1.00         2
# weighted avg       1.00      1.00      1.00         2
