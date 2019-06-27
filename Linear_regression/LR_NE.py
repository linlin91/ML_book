"""Linear regression using normal equation"""

from sklearn.linear_model import LinearRegression
import numpy as np

# make up some training data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# make up some test data
X_new = np.array([[0], [2]])


################ write from sketch ###############

X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance

# fit model
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
# model parameters
print(theta_best)
# model prediction
X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance
y_predict = X_new_b.dot(theta_best)


############## using sklearn ##################

# fit model
lin_reg = LinearRegression()
lin_reg.fit(X, y)
# model parameters
print(lin_reg.intercept_, lin_reg.coef_)
# model prediction
print(lin_reg.predict(X_new))
# The LinearRegression class is based on the scipy.linalg.lstsq() function (the name stands for "least squares")

