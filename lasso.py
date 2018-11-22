import numpy as np
import pandas as pd
import math
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

saveName = "Lasso"
data_file = "data/clean_train.csv"
alphas = [0.01, 0.1, 1, 10, 100, 1000, 10000]

def cv_estimator(X, y):
    reg = LassoCV(cv=10, verbose=0, alphas=alphas, max_iter=100000).fit(X,y)
    print("alpha:",reg.alpha_)
    # print("coefficients:", reg.coef_)
    return reg.alpha_

def lasso_regression(X_train, X_validate, y_train, y_validate, alpha):
    reg = Lasso(alpha=alpha, max_iter=100000)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_validate)
    mse = mean_squared_error(y_validate,y_pred)
    print("RMSE:", math.sqrt(mse))

data = pd.read_csv(data_file)
X = data.drop('SalePrice', axis=1).values
y = data['SalePrice'].values
X_train, X_validate, y_train, y_validate = train_test_split(X,y,test_size=0.2)
alpha = cv_estimator(X_train,y_train)
lasso_regression(X_train, X_validate, y_train, y_validate, alpha)