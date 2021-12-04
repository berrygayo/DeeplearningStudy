### 가옹 체크할 것 
# 사용 변수가 적은것 같음 ... > 전체변수 한번 더 정리해서 넣으면 점수 높아질 것 같음 
# 각 모델별 params가 뜻하는게 무엇인지 정리 

# 따라서 앞으로 해볼만한 연구는 
# 1. 최대한 많은 변수 넣어서 동일한 코드에 적용해보기. 
# 2.현재 진행했던 모델은 3개인데 더 많은 모델로 분석 돌려보고, 결과를 보고 상위 n 개 앙상블 모델 다시 진행 
# 3. 최종 코드 class 로 정리 및 .py 파일로 저장(?)

import numpy as numpy
import pandas as pd 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 

###################################################   loading_data      ################################################
train = pd.read_csv('./bicycle_trainingset_211118_t.csv')

train_name = ['year','rainyday', 'temp_scaled', 'h_0',
              'h_1', 'h_2', 'h_3', 'h_4', 'h_5', 'h_6', 'h_7', 'h_8', 'h_9', 'h_10',
              'h_11', 'h_12', 'h_13', 'h_14', 'h_15', 'h_16', 'h_17', 'h_18', 'h_19',
              'h_20', 'h_21', 'h_22', 'h_23','hw_0', 'hw_1', 'hw_2', 'hw_3', 'hw_4', 'hw_5',
              'hw_6', 'hw_7', 'hw_8', 'hw_9', 'hw_10', 'hw_11', 'hw_12', 'hw_13',
              'hw_14', 'hw_15', 'hw_16', 'hw_17', 'hw_18', 'hw_19', 'hw_20', 'hw_21',
              'hw_22', 'hw_23', 'count']

X_train_name = ['year','rainyday', 'temp_scaled', 'h_0',
                'h_1', 'h_2', 'h_3', 'h_4', 'h_5', 'h_6', 'h_7', 'h_8', 'h_9', 'h_10',
                'h_11', 'h_12', 'h_13', 'h_14', 'h_15', 'h_16', 'h_17', 'h_18', 'h_19',
                'h_20', 'h_21', 'h_22', 'h_23','hw_0', 'hw_1', 'hw_2', 'hw_3', 'hw_4', 'hw_5',
                'hw_6', 'hw_7', 'hw_8', 'hw_9', 'hw_10', 'hw_11', 'hw_12', 'hw_13',
                'hw_14', 'hw_15', 'hw_16', 'hw_17', 'hw_18', 'hw_19', 'hw_20', 'hw_21',
                'hw_22', 'hw_23']

Y_train_name = 'count'

train = train.loc[:, train_name]
X = train.loc[:, X_train_name]
target = train[Y_train_name]
target_log = np.log1p(target)

###################################################       split data       ################################################

print(">>> 현재 y값은 로그값이 취한 값으로 실측값을 확인하고 싶을땐 exponential 필요합니다. ") 
X_train, X_valid, y_train, y_valid = train_test_split(X, target_log, test_size = 0.25, random_state = 123)

###################################################       Scorer       ################################################

# RMSLE Scorer
def rmsle(y,y_,convertExp=True):
    if convertExp:
        print("지수화를 실행합니다.")
        y = np.exp(y),
        y_ = np.exp(y_)
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))


###################################################      Modeling 1     ################################################
### Score train, valid ###
### 파라미터 튜닝 전 ###
### RF 0.275, 0.415 ###
### LR 0.391, 0.397 ###
### GB 0.451, 0.467 ###
############################

# Randomforest Model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 

RF_reg= RandomForestRegressor(random_state=0)
RF_reg.fit(X_train, y_train)

preds_train = RF_reg.predict(X_train)
preds_valid = RF_reg.predict(X_valid)
print("Randomforest Train Score is: ",rmsle(np.exp(y_train), np.exp(preds_train),False) )
print("Randomforest Valid Score is: ",rmsle(np.exp(y_valid), np.exp(preds_valid),False) )
# 지수화 상태로 넣고싶지 않다면
#print("Randomforest Train Score is: ",rmsle(y_train, preds_train) )
#print("Randomforest Valid Score is: ",rmsle(y_valid, preds_valid) )

# Linear Regression Model 
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import metrics 

LM_reg = LinearRegression()
LM_reg.fit(X_train, y_train) 

preds_train = LM_reg.predict(X_train)
preds_valid = LM_reg.predict(X_valid)
print("Linear Regression Train Score is: ",rmsle(np.exp(y_train), np.exp(preds_train),False) )
print("Linear Regression Valid Score is: ",rmsle(np.exp(y_valid), np.exp(preds_valid),False) )

# Gradient Boost Model
from sklearn.ensemble import GradientBoostingRegressor 
gb_reg = GradientBoostingRegressor()
gb_reg.fit(X_train, y_train)

preds_train = gb_reg.predict(X_train)
preds_valid = gb_reg.predict(X_valid)
print("Gradient Boost Train Score is: ",rmsle(np.exp(y_train), np.exp(preds_train),False) )
print("Gradient Boost Valid Score is: ",rmsle(np.exp(y_valid), np.exp(preds_valid),False) )


###################################################      param tuning     ################################################
### RamdomForest best param is :  {'n_estimators': 911, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_depth': 80}
### GradientBoosting best param is :  {'alpha': 0.1, 'learning_rate': 0.3, 'max_depth': 3} 


%%time 
# lr은 param tuning 없이 바로 진행
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html# 
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.pipeline import Pipeline, make_pipeline 
from sklearn.model_selection import RandomizedSearchCV

param_grid_rf = [{
                  'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 1000, num = 10)], # 트리 수 
                  'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)], # 트리의 최대 깊이
                  'min_samples_split': [2, 5, 10], # 내부 노드 분할하는데 필요한 최소샘플 수 
                  'min_samples_leaf': [1, 2, 4], # 리프 노드에 있어야하는 최소 샘플 수 
                }]

param_grid_gb = [{'max_depth':[3,5],
             'learning_rate':[0.1, 0.01, 0.3],
                'alpha':[0.5, 0.1, 0.9]}]

rf_random = RandomizedSearchCV(estimator = RF_reg, param_distributions = param_grid_rf, n_iter = 100, cv = 5, verbose = 2, random_state = 0, n_jobs= -1)
rf_random.fit(X_train, y_train)

gb_random = GridSearchCV(estimator = gb_reg, param_grid = param_grid_gb, cv=10, n_jobs=-1, verbose=2)
gb_random.fit(X_train, y_train)

# rf 
print("RamdomForest best param is : ", grid_rf.best_params_)
print("RamdomForest best score is : ",grid_rf.best_score_)
# gb 
print("GradientBoosting best param is : ", grid_gb.best_params_)
print("GradientBoosting best score is : ", grid_gb.best_score_)


###################################################      Modeling 2     ################################################
### Score train, valid ###
### 파라미터 튜닝 후 ###
### RF 0.317, 0.391 ###
### LR 0.391, 0.397 ###
### GB 0.354, 0.376 ###
############################

# Randomforest Model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 

RF_reg= RandomForestRegressor(n_estimators= 911, min_samples_split= 10, min_samples_leaf= 2, max_depth= 80)
RF_reg.fit(X_train, y_train)

preds_train = RF_reg.predict(X_train)
preds_valid = RF_reg.predict(X_valid)
print("Randomforest Train Score is: ",rmsle(np.exp(y_train), np.exp(preds_train),False) )
print("Randomforest Valid Score is: ",rmsle(np.exp(y_valid), np.exp(preds_valid),False) )

# Linear Regression Model 
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import metrics 

LM_reg = LinearRegression()
LM_reg.fit(X_train, y_train) 

preds_train = LM_reg.predict(X_train)
preds_valid = LM_reg.predict(X_valid)
print("Linear Regression Train Score is: ",rmsle(np.exp(y_train), np.exp(preds_train),False) )
print("Linear Regression Valid Score is: ",rmsle(np.exp(y_valid), np.exp(preds_valid),False) )

# Gradient Boost Model
from sklearn.ensemble import GradientBoostingRegressor 
gb_reg = GradientBoostingRegressor(alpha= 0.1, learning_rate= 0.3, max_depth= 3)
gb_reg.fit(X_train, y_train)

preds_train = gb_reg.predict(X_train)
preds_valid = gb_reg.predict(X_valid)
print("Gradient Boost Train Score is: ",rmsle(np.exp(y_train), np.exp(preds_train),False) )
print("Gradient Boost Valid Score is: ",rmsle(np.exp(y_valid), np.exp(preds_valid),False) )


###################################################      Voting Regressor     ################################################
### voting score ### 
### 0.340, 0.372 ### 
####################

from sklearn.ensemble import VotingRegressor
model_vote = VotingRegressor([ ('LinearRegression',LM_reg),('Randomforest',RF_reg),('GradientBoosting',gb_reg) ])
model_vote.fit(X_train, y_train)

preds_train = model_vote.predict(X_train)
preds_valid = model_vote.predict(X_valid)
print("Voting Regressor Train Score is: ",rmsle(np.exp(y_train), np.exp(preds_train),False) )
print("Voting Regressor Valid Score is: ",rmsle(np.exp(y_valid), np.exp(preds_valid),False) )

###################################################      model save    ################################################
import pickle 
from joblib import dump, load

dump(model_vote, 'voting_rf_lr_gb.pkl') 

###################################################      model loading    ################################################
clf_from_joblib = load('voting_rf_lr_gb.pkl') 
clf_from_joblib.predict(X_valid)