"""Linear regression using batch gradient descent"""

import numpy as np

# make up some training data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X] # add x0 = 1 to each instance

# make up some test data
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance


################ write from sketch ###############

# fit model
eta = 0.1 # learning rate
n_iterations = 1000
m = X.shape[0]
theta = np.random.randn(2,1) # random initialization
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y) # X_b.dot(theta) is the y hat
    theta = theta - eta * gradients
# model parameters
print(theta)
# model prediction
print(X_new_b.dot(theta))


############## cannot find package in sklearn ##################