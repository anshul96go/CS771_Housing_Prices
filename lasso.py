import numpy as np
import pandas as pd
import math
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

saveName = "lasso"
data_file = "data/clean_train.csv"
alphas = [0.01, 0.1, 1, 10, 100, 1000, 10000]

imagedir = 'images/'
if not os.path.exists(imagedir):
    os.makedirs(imagedir)

def cv_estimator(X, y):
    reg = LassoCV(cv=10, verbose=0, alphas=alphas, max_iter=100000).fit(X,y)
    print("alpha:",reg.alpha_)
    return reg.alpha_

def lasso_regression(X_train, X_validate, y_train, y_validate, alpha):
    reg = Lasso(alpha=alpha, max_iter=100000)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_validate)
    mse = mean_squared_error(y_validate,y_pred)
    rmse = math.sqrt(mse)
    print("RMSE:", rmse)
    return reg.coef_, y_pred, rmse

def custom_plot(column, y_pred, y_validate, x):
    plt.figure(figsize=[15,10])
    plt.scatter(x, y_pred, color='r', label="Predicted Points")
    plt.scatter(x, y_validate, color='g', label="Sales Price")
    plt.legend()
    plt.title(saveName+' '+column)
    plt.savefig(imagedir+saveName+'_'+column+'.png')
    plt.close()

data = pd.read_csv(data_file)
X = data.drop('SalePrice', axis=1).values
columns = data.drop('SalePrice', axis=1).columns.values
y = data['SalePrice'].values
X_train, X_validate, y_train, y_validate = train_test_split(X,y,test_size=0.2)
alpha = cv_estimator(X_train,y_train)
w, y_pred, rmse = lasso_regression(X_train, X_validate, y_train, y_validate, alpha)
print(columns.shape, w.shape, X_validate.shape)
selected_columns = []

count = 0
for i in range(len(w)):
    if abs(int(w[i])) > 0:
        #custom_plot(columns[i], y_pred, y_validate, X_validate[:,i])
        selected_columns.append(columns[i])
        count+=1
print(count)

with open(saveName+'_'+'results.txt', 'w') as f:
    f.write('Number of features: ' + str(w.shape[0]) + '\n')
    f.write('Features with non zero weight: ' + str(count) + '\n')
    f.write('Best regularization parameter: ' + str(alpha) + '\n')
    f.write('Root Mean Squared Error: ' + str(rmse) + '\n')
    f.write('Selected Columns:\n')
    for column in selected_columns:
        f.write(column + '\n')