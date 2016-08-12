# Author: Nils A. Treiber <nils.andre.treiber@uni-oldenburg.de>
# Oliver Kramer <okramer@icsi.berkeley.com>
# Jendrik Poloczek <jendrik.poloczek@madewithtea.com>
# Justin P. Heinermann <justin.heinermann@uni-oldenburg.de>
# Stefan Oehmcke <stefan.oehmcke@uni-oldenburg.de>
# License: BSD 3 clause

import math
import matplotlib.pyplot as plt
from numpy import zeros, float32
import numpy as np
from windml.datasets.nrel import NREL
from windml.mapping.power_mapping import PowerMapping
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor


# get windpark and corresponding target. forecast is for the target turbine
tb_names = ['tehachapi', 'cheyenne', 'palmsprings', 'reno', 'lasvegas', 'hesperia', 'lancaster', 'yuccavalley', 'vantage', 'casper']
turbine_name = tb_names[0]  
park_id = NREL.park_id[turbine_name]
windpark = NREL().get_windpark(park_id, 3, 2004, 2005)
target = windpark.get_target()

# use power mapping for pattern-label mapping.
feature_window, horizon = 3,3
mapping = PowerMapping()
X = mapping.get_features_park(windpark, feature_window, horizon)
y = mapping.get_labels_turbine(target, feature_window, horizon)

# train roughly for the year  2004, 2005 test roughly for the year 2006.
train_to = [0]
data_bags =1
for i in range(data_bags): 
    train_to.append(int(math.floor(len(X) * (18/24.0)*(i+1)/data_bags)))
test_to = len(X)

# train and test only every fifth pattern, for performance.
train_step, test_step = 5, 5

X_train = []
y_train = []
for i in range(data_bags):
    X_train.append(X[train_to[i]:train_to[i+1]:train_step])
    y_train.append(y[train_to[i]:train_to[i+1]:train_step])
X_test=X[train_to[-1]:test_to:test_step]
y_test=y[train_to[-1]:test_to:test_step]


class BA_ensemble(object):
    '''
    Variable 'estimators' can be seen as a list of models. Each element of the list 
    is defined as: ['Model_Name', [Model Paramters]]
 
    In all there can be following 7 types of ML models with these parameters: 
     1. 'SVR': kernal = 'rbf', degree = 3, gamma = 1e-4, C = 100, epsilon = 0.1
     2. 'RF': max_depth = 30, n_estimators = 10, n_jobs = -1, random_state = 0
     3. 'KNN': n_neighbors = 5, weights = 'uniform', leaf_size = 30
     4. 'Ridge': alpha = 0.5
     5. 'Bayesian': alpha_1 = 1e-6, alpha_2 = 1e-6
     6. 'LR': fit_intercept =True , n_jobs = -1, normalize =False
     7. 'GBR': learning_rate = 0.1, n_estimators = 100, max_depth = 3, loss='ls'
    '''
    def __init__(self, train_x, train_y, estimators):
        '''
        Creates and Trains all the models
        '''
        self.reg = []        
        for d in range(len(train_x)):
            self.X_train = train_x[d] 
            self.Y_train = train_y[d]
            for i in range(len(estimators)):
                self.reg.append(Estimator(info=estimators[i]))
                if estimators[i][0] == 'SVR':
                    self.reg[-1].train(SVR(kernel=estimators[i][1][0], degree=estimators[i][1][1], gamma = estimators[i][1][2] , C = estimators[i][1][3], epsilon=estimators[i][1][4]).fit(self.X_train,self.Y_train))
                elif estimators[i][0] == 'RF':
                    self.reg[-1].train(RandomForestRegressor(max_depth=estimators[i][1][0], n_estimators=estimators[i][1][1], n_jobs=estimators[i][1][2], random_state=estimators[i][1][3]).fit(self.X_train,self.Y_train))
                elif estimators[i][0] == 'KNN':
                    self.reg[-1].train(KNeighborsRegressor(n_neighbors=estimators[i][1][0], weights=estimators[i][1][1], leaf_size=estimators[i][1][2]).fit(self.X_train,self.Y_train))
                elif estimators[i][0] == 'Ridge':
                    self.reg[-1].train(linear_model.Ridge(alpha = estimators[i][1][0]).fit(self.X_train,self.Y_train))
                elif estimators[i][0] == 'Bayesian':
                    self.reg[-1].train(linear_model.BayesianRidge(alpha_1 = estimators[i][1][0], alpha_2 = estimators[i][1][1]).fit(self.X_train,self.Y_train))
                elif estimators[i][0] == 'LR':
                    self.reg[-1].train(linear_model.LinearRegression(fit_intercept = estimators[i][1][0], n_jobs = estimators[i][1][1], normalize = estimators[i][1][2]).fit(self.X_train,self.Y_train))
                elif estimators[i][0] == 'GBR':
                    self.reg[-1].train(GradientBoostingRegressor(learning_rate = estimators[i][1][0], n_estimators = estimators[i][1][1], max_depth = estimators[i][1][2],  loss=estimators[i][1][3]).fit(self.X_train,self.Y_train))
    def predict(self, test_x, test_y):
        '''
        Predicts/Forecasts using all the models
        '''
        self.X_test = test_x 
        self.Y_test = test_y
        self.y_hats = []        
        for i in range(len(self.reg)):
            self.reg[i].predictions(self.reg[i].regressor.predict(self.X_test))
            # Mean Square Error
            self.reg[i].PI_update('mse', mean_squared_error(self.Y_test, self.reg[i].y_hat))
    def aggregate(self, alpha = []):
        '''
        Ensembles (Weighted Bagging/Bootstrap-Aggregation) all the predictions  
        '''
        if len(alpha) < len(self.reg):
            for i in range(len(self.reg)-len(alpha)):
                alpha.append(1)
        self.alpha = alpha
        self.y_hat = []
        for i in range(len(self.X_test)):
            self.y_hat.append(0)            
            for j in range(len(self.reg)):       
                self.y_hat[-1] = self.y_hat[-1] + self.alpha[j] * self.reg[j].y_hat[i]
            self.y_hat[-1] = self.y_hat[-1]/(float(sum(self.alpha)))
        # Mean Square Error
        self.mse = mean_squared_error(self.Y_test, self.y_hat)

            
class Estimator(object):
    '''
    'Estimator' object is the indivdual ML model.
    self.info : Information about type of ML model and its parameter values    
    self.regressor : This variable stores the sklearn based trained model of type defined in self.info
    self.y_hat : Stores the predictions made by the model
    self.PI : can be used to store different performance indices for the predictions
    '''    
    def __init__(self, info):
        self.info = info
        self.PI = []
    def train(self, trainer):
        '''
        Stores the regression model
        '''
        self.regressor = trainer
    def predictions(self, haty):
        '''
        Stores the predicted/forecasted values 
        '''
        self.y_hat = haty
    def PI_update(self, PI_type,  value):
        '''
        Stores the performance indices in th self.PI list
        '''
        self.PI.append([PI_type, value])
    

models= []
models.append(['RF', [30, 10, -1, 2]]) 
models.append(['Ridge',[0.2]])
models.append(['Bayesian',[1e-6,1e-6]])
models.append(['GBR',[0.1, 100, 3, 'ls']])
models.append(['KNN',[5, 'uniform', 30]])

bag = BA_ensemble(X_train, y_train, models)
bag.predict(X_test, y_test)
bag.aggregate([1,1,1,1,1])


for i in range(len(bag.reg)):
    print(bag.reg[i].PI)
print(bag.mse)


with plt.style.context("fivethirtyeight"):
    figure = plt.figure(figsize=(8, 5))
    time = range(0, len(bag.y_hat))
    plt.plot(time, y_test, label="Measurement")
    plt.plot(time, bag.y_hat, label="Prediction")
    plt.xlim([1000, 1400])
    plt.ylim([-5, 50])
    plt.xlabel("Time Steps")
    plt.ylabel("Wind Power [MW]")
    plt.legend()
    plt.title(turbine_name)
    plt.tight_layout()
    plt.savefig('Results/'+turbine_name+'_pred.jpeg')
    plt.close()


# MSE bar plot
fig, ax = plt.subplots()
ind = np.arange(1, len(bag.alpha)+2)
bars_data = []
bars_xtick = []
for i in range(len(bag.alpha)):
    bars_data.append(bag.reg[i].PI[0][1])
    bars_xtick.append(bag.reg[i].info[0])    
bars_data.append(bag.mse)
bars_xtick.append('BA')
bars = plt.bar(ind, bars_data, align="center")
for i in range(len(bag.alpha)):
    bars[i].set_facecolor('r')          
bars[-1].set_facecolor('b')    
ax.set_xticks(ind[0:len(bag.alpha)+1])
ax.set_xticklabels(bars_xtick)
ax.set_ylim([int(min(bars_data)), int(max(bars_data)+1)])
ax.set_ylabel('Mean Square Error')
ax.set_title(turbine_name)
plt.savefig('Results/'+turbine_name+'_mse_bar.jpeg')
plt.close()
