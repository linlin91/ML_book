"""Linear regression using stochastic gradient descent"""

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
n_epochs = 50
t0, t1 = 5, 50  # learning schedule hyperparameters
m = X.shape[0]

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2,1)  # random initialization

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients

# model parameters
print(theta)
# model prediction
print(X_new_b.dot(theta))


############## using sklearn ##################


from sklearn.linear_model import SGDRegressor

# fit model
sgd_reg = SGDRegressor(max_iter=50, tol=-np.infty, penalty=None, eta0=0.1, random_state=42) # defualts to optimize the squared error cost function
                                                                                            # epoch (max_iter) = 50;
                                                                                            # learning rate (eta0) starting value 0.1, learning schedule default, different from the above one
                                                                                            # don't use any regularization (penalty=None)
sgd_reg.fit(X, y.ravel()) # .ravel() Return a contiguous flattened array.
# model parameters
print(sgd_reg.intercept_, sgd_reg.coef_)
# model prediction
print(sgd_reg.predict(X_new))