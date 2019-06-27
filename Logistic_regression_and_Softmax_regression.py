"""Logistic regression and softmax regression"""
import numpy as np
from sklearn import datasets

# get data
iris = datasets.load_iris()
X = iris["data"][:, 3:]  # petal width
y = (iris["target"] == 2).astype(np.int)  # 1 if Iris-Virginica, else 0
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)

# Logistic regression
# fit model
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver="liblinear", random_state=42)
log_reg.fit(X, y)

# model prediction
y_proba = log_reg.predict_proba(X_new)

#Softmax regression
# get data
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]
x0, x1 = np.meshgrid(
        np.linspace(0, 8, 500).reshape(-1, 1),
        np.linspace(0, 3.5, 200).reshape(-1, 1),
    )
X_new = np.c_[x0.ravel(), x1.ravel()]

# fit model
softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10, random_state=42)
softmax_reg.fit(X, y)

# model prediction
y_proba = softmax_reg.predict_proba(X_new)
y_predict = softmax_reg.predict(X_new)