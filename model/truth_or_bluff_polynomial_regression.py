# Importing libraries:
import matplotlib.pyplot as plt  # Used for plotting data
import pandas as pd  # Used for data manipulation

# Importing modules from the scikit-learn library:
from sklearn.linear_model import LinearRegression  # Linear Regression model
from sklearn.preprocessing import PolynomialFeatures  # Polynomial Regression features

# Initializing variables:
dataset = pd.read_csv("../dataset/Position_Salaries.csv")  # Load dataset from CSV file
x = dataset.iloc[:, 1:-1].values  # Features (independent variables)
y = dataset.iloc[:, -1].values  # Dependent variable (target)

# Initializing linear regression model and fitting data:
regressor = LinearRegression()
regressor.fit(x, y)

# Printing features and dependent variable:
print("Features (x):")
print(x)
print("Dependent Variable (y):")
print(y)

# Initializing polynomial regression model and fitting data:

degree = 4  # Degree of polynomial features
pf = PolynomialFeatures(degree=degree) # Initialised variable for the polynomial regression model
x_poly = pf.fit_transform(x) # Fit and transform the features and store result in new variable
lin_reg = LinearRegression() # Initialise new variable to a new linear regression instance
lin_reg.fit(x_poly, y) # Fit the new linear regression instance on the polynomial features and dependent variable

# Plotting points using linear regression:
plt.scatter(x, y, color="red")
plt.plot(x, regressor.predict(x), color="blue")
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Plotting points using polynomial regression:
plt.scatter(x, y, color="red")
plt.plot(x, lin_reg.predict(x_poly), color="blue")
plt.title(f"Truth or Bluff (Polynomial Regression - Degree {degree})")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Predicting salary using linear regression for a new position level:
new_position_level = 6.5
predicted_salary_linear = regressor.predict([[new_position_level]])
print(f"Predicted Salary (Linear Regression) for Position Level {new_position_level}: ${predicted_salary_linear[0]:,.2f}")

# Predicting salary using polynomial regression for a new position level:
predicted_salary_poly = lin_reg.predict(pf.transform([[new_position_level]]))
print(f"Predicted Salary (Polynomial Regression - Degree {degree}) for Position Level {new_position_level}: ${predicted_salary_poly[0]:,.2f}")
