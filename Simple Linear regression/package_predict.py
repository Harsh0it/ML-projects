import numpy as np
import pandas as pd


df=pd.read_csv( r'C:\Users\HP\Downloads\placement.csv')
print(df.head())

X=df.iloc[:,0:1]
y=df.iloc[:,-1]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(X_train, y_train)

cgpa_input = float(input("Enter your CGPA: "))


predicted_package = lr.predict([[cgpa_input]]) 

print(f"Predicted Package: ₹{predicted_package[0]:.2f} LPA")