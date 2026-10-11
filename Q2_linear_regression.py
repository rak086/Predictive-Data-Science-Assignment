# Question 2: Simple Linear Regression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample Data: Years of Experience vs Salary (in thousands)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([30, 35, 45, 50, 60, 68, 75, 82, 90, 105])

# Create and fit model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

print("--- Simple Linear Regression Output ---")
print(f"Slope (Coefficient): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# Example prediction for 5.5 years of experience
new_exp = np.array([[5.5]])
predicted_salary = model.predict(new_exp)
print(f"Predicted Salary for 5.5 years of experience: ${predicted_salary[0]:.2f}k")
