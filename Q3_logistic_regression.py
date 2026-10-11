# Question 3: Logistic Regression

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample Data: Age and Estimated Salary to predict purchase (0 = No, 1 = Yes)
X = np.array([
    [22, 20000], [25, 34000], [47, 45000], [52, 60000], [46, 82000],
    [56, 133000], [41, 72000], [35, 45000], [28, 30000], [49, 86000]
])
y = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 1])

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model fitting
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("--- Logistic Regression Output ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
