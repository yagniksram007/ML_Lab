#Implement and demonstrate the Find-S algorithm for finding the most specific hypothesis


import numpy as np

class FindS:
    def __init__(self, num_features):
        self.num_features = num_features
        self.hypothesis = np.array(['0'] * num_features)

    def fit(self, X, y):
        for i in range(len(X)):
            if y[i] == 1:  # Positive instance
                for j in range(self.num_features):
                    if self.hypothesis[j] == '0':
                        self.hypothesis[j] = X[i][j]
                    elif self.hypothesis[j] != X[i][j]:
                        self.hypothesis[j] = '?'

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            positive = True
            for j in range(self.num_features):
                if self.hypothesis[j] != '?' and self.hypothesis[j] != X[i][j]:
                    positive = False
                    break
            predictions.append(positive)
        return predictions

# Example usage:
X_train = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change']
]
y_train = [1, 1, 0, 1]  # 1 for positive instances, 0 for negative

model = FindS(num_features=len(X_train[0]))
model.fit(X_train, y_train)

print("Most specific hypothesis:", model.hypothesis)

X_test = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'],
    ['Sunny', 'Cold', 'High', 'Strong', 'Warm', 'Same']
]

predictions = model.predict(X_test)
print("Predictions:", predictions)
