import numpy as np
import pandas as pd
import math
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score

saveName = "lasso"
data_file = "data/clean_train.csv"
selected_features_file = "data/selected_features.csv"
alphas = [0.01, 0.1, 1, 10, 100, 1000, 10000]

imagedir = 'plots/'
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
    r2 = r2_score(y_validate, y_pred)
    rmse = math.sqrt(mean_squared_error(y_validate,y_pred))
    print("RMSE:", rmse)
    return reg.coef_, y_pred, rmse, r2

def custom_plot(name, y_pred, y_validate, x):
    plt.figure(figsize=[15,10])
    plt.scatter(x, y_pred, color='r', label="Predicted Points")
    plt.scatter(x, y_validate, color='g', label="Sales Price")
    plt.legend()
    plt.xticks(x, [])
    plt.grid(True, axis='both', linewidth=0.2)
    plt.title(saveName+' '+name)
    plt.savefig(imagedir+saveName+'_'+name+'.png')
    plt.close()

data = pd.read_csv(data_file, index_col='Id')
X = data.drop('SalePrice', axis=1).values
columns = data.drop('SalePrice', axis=1).columns.values
y = data['SalePrice'].values
X_train, X_validate, y_train, y_validate = train_test_split(X,y,test_size=0.2)
alpha = cv_estimator(X_train,y_train)
w, y_pred, rmse, r2 = lasso_regression(X_train, X_validate, y_train, y_validate, alpha)
# print(columns.shape, w.shape, X_validate.shape)
selected_columns = []
selected_features_columns = pd.read_csv(selected_features_file).columns.values
intersection_columns = []

for i in range(len(w)):
    if abs(int(w[i])) > 0:
        selected_columns.append(columns[i])
        if columns[i] in selected_features_columns:
            intersection_columns.append(columns[i])
print(len(selected_columns))
print(len(intersection_columns))

custom_plot('Prediction', y_pred, y_validate, range(1,len(y_pred)+1))
custom_plot('Prediction_Magnified', y_pred[20:50], y_validate[20:50], range(1,len(y_pred[20:50])+1))

with open(saveName+'_'+'results.txt', 'w') as f:
    f.write('Number of features: ' + str(w.shape[0]) + '\n')
    f.write('Number of features with non zero weight: ' + str(len(selected_columns)) + '\n')
    f.write('Number of features with non zero weight in selected features: ' + str(len(intersection_columns)) + '\n')
    f.write('Best regularization parameter: ' + str(alpha) + '\n')
    f.write('R2 Score: ' + str(r2) + '\n')
    f.write('Root Mean Squared Error: ' + str(rmse) + '\n\n')
    f.write('Features with non zero weight:\n')
    for column in selected_columns:
        f.write(column + '\n')
    f.write('\nFeatures with non zero weights in selected features:\n')
    for column in intersection_columns:
        f.write(column + '\n')