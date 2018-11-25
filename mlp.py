import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import math

imagedir = 'plots/'
if not os.path.exists(imagedir):
    os.makedirs(imagedir)

saveName = "mlp"
data_file = "data/selected_features.csv"
alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
hidden_layer_sizes = [[36], [48,24], [54,36,18]]

def grid_search(X_train, y_train):
    reg = MLPRegressor(max_iter=10000, early_stopping=False, solver='adam', activation='relu')
    parameters = {'alpha':alphas, 'hidden_layer_sizes':hidden_layer_sizes[1:2]}
    cv_estimator = GridSearchCV(reg, param_grid=parameters, cv=5, verbose=1, n_jobs=-1)
    cv_estimator.fit(X_train, y_train)
    print(cv_estimator.best_params_)
    return cv_estimator.best_params_

def nn_regression(X_train, X_validate, y_train, y_validate, params):
    reg = MLPRegressor(hidden_layer_sizes=params['hidden_layer_sizes'],
                        max_iter=10000, alpha=params['alpha'],
                        early_stopping=False,
                        solver='adam',
                        activation='relu')
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_validate)
    mse = mean_squared_error(y_validate,y_pred)
    rmse = math.sqrt(mse)
    print("RMSE:", rmse)
    return y_pred, rmse

def custom_plot(name, y_pred, y_validate, x):
    plt.figure(figsize=[15,10])
    plt.scatter(x, y_pred, color='r', label="Predicted Points")
    plt.scatter(x, y_validate, color='g', label="Sales Price")
    plt.legend()
    plt.xticks(x, [])
    plt.grid(True, axis='both', linewidth=0.1)
    plt.title(saveName+' '+name)
    plt.savefig(imagedir+saveName+'_'+name+'.png')
    plt.close()

data = pd.read_csv(data_file)
columns = data.drop('SalePrice', axis=1).columns.values
X = data.drop('SalePrice', axis=1).values
y = data['SalePrice'].values
X_train, X_validate, y_train, y_validate = train_test_split(X,y,test_size=0.2)
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_validate = scaler.transform(X_validate)
# best_params = grid_search(X_train, y_train)
best_params = {'alpha':alphas[2], 'hidden_layer_sizes':hidden_layer_sizes[1]}
y_pred, rmse = nn_regression(X_train, X_validate, y_train, y_validate, best_params)

custom_plot('Prediction', y_pred, y_validate, range(1,len(y_pred)+1))
custom_plot('Prediction_Magnified', y_pred[20:50], y_validate[20:50], range(1,len(y_pred[20:50])+1))

with open(saveName+'_'+'results.txt', 'w') as f:
    f.write('Number of features: ' + str(X.shape[1]) + '\n')
    f.write('Best regularization parameter: ' + str(best_params['alpha']) + '\n')
    f.write('Hidden Layer Sizes: ')
    for num in best_params['hidden_layer_sizes']:
        f.write(str(num)+' ')
    f.write('\n')
    f.write('Root Mean Squared Error: ' + str(rmse) + '\n')