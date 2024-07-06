# Importing the libraries

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Initializing variables:
dataset = pd.read_csv("./dataset/Position_Salaries.csv")  # Load dataset from CSV file
x = dataset.iloc[:, 1:-1].values  # Features (independent variables)
y = dataset.iloc[:, -1].values  # Dependent variable (target)

# Printing the features and dependent variable
print(x)
print(y)

# Changing the shape of the dependent variable to have each result in different rows (transpose)
y = y.reshape(len(y), 1)
 
# See the result of transpose.
print(y)

# From here, we can now apply feature scaling on both x and y.

y = y.reshape(len(y), 1)
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

print(x)
print(y)

# Specifying that we want to use the kernel "Radial Basis Function"/RBF.
regressor = SVR(kernel="rbf")
regressor.fit(x, y)

"""
Firstly, we predict the transformation using SVR on our problem statement level (6.5)
Then as we applied feature scaling on the dependent varaible, we'll need to retrieve its original value back.
This is what "sc_y.inverse_transform" handles. 
Finally, we reshape it to inverse the shape it originally had.
"""

sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1,1))

plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color="red")
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x).reshape(-1,1)), color="blue")
plt.title("Truth or Bluff (SVR)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()