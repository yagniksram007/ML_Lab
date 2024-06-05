# Demonstrate the working of SVM classifier for a suitable dataset

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

#loading the data from the csv file to pandas dataframe
diabetes_data = pd.read_csv('D:/My Files/Engineering/6th Sem/Machine Learning Lab/Dataset/diabetes.csv')


diabetes_data['Outcome'].value_counts()

# separating the features and target
features = diabetes_data.drop(columns='Outcome', axis=1)
target = diabetes_data['Outcome']

# Data Standardisation
scaler = StandardScaler()
scaler.fit(features)
standardized_data = scaler.transform(features)

features = standardized_data
target = diabetes_data['Outcome']

# Preprocess the data
X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2, random_state = 42)

# Train the model
classifier = svm.SVC(kernel = 'linear')
classifier.fit(X_train, Y_train)

# Model Evaluation
# accuracy on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print('Accuracy score on training data = ', training_data_accuracy)

# accuracy on training data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print('Accuracy score on test data = ', test_data_accuracy)

# Testing phase
input_data = (5,166,72,19,175,25.8,0.587,51)

# input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardizing the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')

else:
  print('The Person is diabetic') 
  
  
# Output:
# Accuracy score on training data =  0.7719869706840391
# Accuracy score on test data =  0.7597402597402597
# [[ 0.3429808   1.41167241  0.14964075 -0.09637905  0.82661621 -0.78595734
#    0.34768723  1.51108316]]
# [1]
# The Person is diabetic  