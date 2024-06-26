import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Prepare the dataset
reviews = [
    ("fun couple love love", "comedy"),
    ("fast furious shoot", "action"),
    ("couple fly fast fun fun", "comedy"),
    ("furious shoot shoot fun", "action"),
    ("fly fast shoot love", "action")
]

# Convert to DataFrame
df = pd.DataFrame(reviews, columns=['text', 'target'])

# Map target labels to numerical values
df['target'] = df['target'].map({'comedy': 0, 'action': 1})

# Split dataset into training and testing sets
X = df['text']
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction (Bag of Words)
vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train the Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))

# Get the unique classes in the test set
unique_classes = sorted(y_test.unique())
target_names = ['comedy' if x == 0 else 'action' for x in unique_classes]

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Classify the new document
D = "fast couple shoot fly"
D_transformed = vectorizer.transform([D])
prediction = clf.predict(D_transformed)

# Map numerical values back to labels
label_map = {0: 'comedy', 1: 'action'}
predicted_label = label_map[prediction[0]]

print(f"The most likely class for '{D}' is '{predicted_label}'.")

# Output:

# Accuracy: 1.0
# Classification Report:
#               precision    recall  f1-score   support

#       action       1.00      1.00      1.00         1

#     accuracy                           1.00         1
#    macro avg       1.00      1.00      1.00         1
# weighted avg       1.00      1.00      1.00         1

# The most likely class for 'fast couple shoot fly' is 'action'.
