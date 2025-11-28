# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset from the CSV file and separate the independent variable (Hours) and dependent variable (Scores).
2. Split the data into training and testing sets, then train a Linear Regression model using the training data.
3. Use the trained model to predict the scores for the test data and compare them with the actual values.
4. Plot the regression line with training and test data, and compute the evaluation metrics: MAE, MSE, and RMSE.

## Program:
```
/*
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


df = pd.read_csv("student_scores.csv")

X = df[['Hours']]
Y = df['Scores']


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
print("Predicted Values:\n", Y_pred)

plt.scatter(X_test, Y_test, color='green', label="Test Data")
plt.plot(X_train, y_pred, color='red', label="Line of Best Fit")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title(Simple Linear Regression")
plt.legend()
plt.show()


mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Jesron Shawn C J 
RegisterNumber:  25012933
*/
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
