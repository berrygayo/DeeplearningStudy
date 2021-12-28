###################################################################

# by Berrygayo, 김가영 
# What We Do ! # 
# Forecast Bike Sharing Demand 
# (데이터 다운로드: https://www.kaggle.com/c/bike-sharing-demand/data)
# datetime - 시간. 연-월-일 시:분:초 로 표현합니다.(ex. 2018-11-01 00:00:00)
# season - 계절. 봄(1), 여름(2), 가을(3), 겨울(4) 순으로 표현합니다.
# holiday - 공휴일. 1이면 공휴일이며, 0이면 공휴일이 아닙니다.
# workingday - 근무일. 1이면 근무일이며, 0이면 근무일이 아닙니다.
# weather - 날씨. 1 ~ 4 사이의 값을 가지며, 구체적으로는 다음과 같습니다.
# 1: 아주 깨끗한 날씨입니다. 또는 아주 약간의 구름이 끼어있습니다.
# 2: 약간의 안개와 구름이 끼어있는 날씨입니다.
# 3: 약간의 눈, 비가 오거나 천둥이 칩니다.
# 4: 아주 많은 비가 오거나 우박이 내립니다.
# temp - 온도. 섭씨(Celsius)로 적혀있습니다.
# atemp - 체감 온도. 마찬가지로 섭씨(Celsius)로 적혀있습니다.
# humidity - 습도.
# windspeed - 풍속.
# casual - 비회원(non-registered)의 자전거 대여량.
# registered - 회원(registered)의 자전거 대여량.
# count - 총 자전거 대여랑. 비회원(casual) + 회원(registered)과 동일합니다.

###################################################################

print("Hi, I am Berry S2")

# Package loading
import numpy as np
import pandas as pd  
import os 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import metrics 
path = 'C:/Users/Berry/Documents/GitHub/DeeplearningStudy/김가영/'
os.chdir(path)

class bicycle_Demand_project:   
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.dataset_preprocessed = None
        self.loaded_data = None
        self.X_train = None 
        self.y_train = None
        self.X_valid = None 
        self.y_valid = None 
        # self.continuous_col = [' ~~연속형 변수 ~~~ ']
        # self.categorical_col = ['~~~ 범주형 변수 ~~~ ']   
        # self.binary_col = ['~~~ 이진 변수 ~~~']
        # # 
        # self.y = ['counts']
        # self.scaler = {}       
        # self.feature_column = {}       
        # self.dataset = None


    def datasetLoad(self,name):
        dataset = pd.read_csv(f"{name}.csv")
        return dataset

    def datasetPreprocessing(self, dataset, saving_name, is_it_train=True):
        dataset['datetime'] = pd.to_datetime(dataset['datetime'])
        dataset['year'] = dataset['datetime'].dt.year
        dataset['month'] = dataset['datetime'].dt.month
        dataset['hour'] = dataset['datetime'].dt.hour
        dataset['dayofweek'] = dataset['datetime'].dt.dayofweek
        dataset['weather'].replace({4 : 3}, inplace = True)
        dataset['windspeed'].replace({0 : np.nan}, inplace = True)
        dataset['windspeed'].fillna(dataset.groupby(['weather', 'season'])['windspeed'].transform('mean'), inplace = True)
        dataset.set_index(dataset['datetime'], drop = True, inplace = True)
        dataset.drop('datetime', axis = 1, inplace = True)         
        
        ## Make Features 
        dataset['rainyday'] = pd.get_dummies(dataset['weather'], prefix='w').drop(['w_1','w_2'], axis=1)
        dataset['ideal'] = dataset[['temp', 'windspeed']].apply(lambda x: (0, 1)[15 <= x['temp'] <= 22  and x['windspeed'] < 30], axis = 1)
        dataset['sticky'] = dataset[['humidity', 'temp']].apply(lambda x: (0, 1)[x['temp'] >= 30 and x['humidity'] >= 60], axis = 1)
        dataset['peak'] = dataset[['hour', 'workingday']].apply(lambda x: [0, 1][(x['workingday'] == 1 and  ( x['hour'] == 8 or 17 <= x['hour'] <= 18 or 12 <= x['hour'] <= 13)) or (x['workingday'] == 0 and  10 <= x['hour'] <= 19)], axis = 1)
        dataset['temp(difference)'] = round(dataset['temp'] - dataset['atemp'], 2)
        dataset['discomfort_index'] = round(1.8*dataset['temp'] - 0.55*(1-dataset['humidity']/100)*(1.8*dataset['temp'] - 26) + 32, 2) 

        ## Scaling : MinMax
        sc_col = ['temp', 'atemp', 'windspeed', 'temp(difference)', 'discomfort_index', 'humidity']
        scaler = MinMaxScaler()
        for i in sc_col :
            dataset[i] = scaler.fit_transform(dataset[[i]])
    
        # Make Dummy 
        ob_col = ['hour', 'year', 'dayofweek', 'season', 'month']
        for i in ob_col :
            dummies = pd.get_dummies(dataset[i], prefix = i)     
            if i == 'year' :
                dataset = pd.concat([dataset, dummies.iloc[:,1]], axis=1)    
            elif i == 'hour':
                dataset = pd.concat([dataset, dummies], axis=1)
                for idx, name in enumerate(dummies):
                    dataset['hw_'+str(idx)] = dummies[name] * dataset['workingday']
                dataset.drop(['hour_4','hw_4'], axis = 1, inplace = True)
            else :
                dummies.drop([i+str('_1')], axis = 1, inplace = True)
                dataset = pd.concat([dataset, dummies], axis=1)
    
        ## 만약 train 데이터라면 count를 log 처리 해줌 + test 데이터에 없는 컬럼 제거
        if is_it_train :
            dataset['log_count'] = np.log1p(dataset['count'])
            # test에 없는 변수 제거
            dataset.drop(['count','casual','registered'], axis = 1, inplace = True)


        ## 불필요한 기존 year, temp, season, dayofweek, hour, month, weather 컬럼 drop
        dataset.drop(['temp', 'year', 'season', 'dayofweek', 'hour', 'month', 'weather'], axis = 1, inplace = True)
        
        self.dataset_preprocessed = dataset

        print(">>> Complete Preprocessing <<< ")  

        # save preprocessed_csv
        self.dataset_preprocessed.to_csv(f'{saving_name}.csv')
        print(">>> Success saving Preprocessing csv <<< ")   

        return self.dataset_preprocessed

    # scoring 
    def rmsle(y,y_,convertExp=True):
        if convertExp:
            print("지수화를 실행합니다.")
            y = np.exp(y),
            y_ = np.exp(y_)
        log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
        log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
        calc = (log1 - log2) ** 2
        return np.sqrt(np.mean(calc))

    # data split
    def data_split(self, train):
        X_feature_cols = ['holiday', 'workingday', 'atemp', 'humidity', 'windspeed',
       'rainyday', 'ideal', 'sticky', 'peak', 'temp(difference)',
       'discomfort_index', 'hour_0', 'hour_1', 'hour_2', 'hour_3', 'hour_5',
       'hour_6', 'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12',
       'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18',
       'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23', 'hw_0', 'hw_1',
       'hw_2', 'hw_3', 'hw_5', 'hw_6', 'hw_7', 'hw_8', 'hw_9', 'hw_10',
       'hw_11', 'hw_12', 'hw_13', 'hw_14', 'hw_15', 'hw_16', 'hw_17', 'hw_18',
       'hw_19', 'hw_20', 'hw_21', 'hw_22', 'hw_23', 'year_2012', 'dayofweek_0',
       'dayofweek_2', 'dayofweek_3', 'dayofweek_4', 'dayofweek_5',
       'dayofweek_6', 'season_2', 'season_3', 'season_4', 'month_2', 'month_3',
       'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9',
       'month_10', 'month_11', 'month_12']
        X_features = train.loc[:,X_feature_cols]
        target = train['log_count']
        print(">>> 현재 y값은 로그(log(x+1))값이 취한 값으로 실측값을 확인하고 싶을땐 exponential 필요합니다. ") 

        # split data 
        X_train, X_valid, y_train, y_valid = train_test_split(X_features, target, test_size = 0.25, random_state = 123)

        return X_train, X_valid, y_train, y_valid

     # 파라미터 튜닝함수  
    def randomforest_autotunes(self, X_train, y_train):
        param_grid_rf = [{
                  'n_estimators': [500,1000,1500], # 트리 수 
                  'max_depth': [9,13,17], # 트리의 최대 깊이
                  'max_features' : ['auto','sqrt']
                }]

        rf_random = GridSearchCV(RandomForestRegressor(), refit=True, param_grid = param_grid_rf, iid=True, cv=5 )
        rf_random.fit(X_train, y_train)
        print('------------------------------------------')
        print('tuning RandomForest')
        print('------------------------------------------')
        print(rf_random.best_params_)
        print(rf_random.best_score_)
        print(rmsle(rf_random.predict(X_train),y_train))

        return rf_random.best_estimator_

    def ridge_autotune(self, X_train, y_train):
        param_grid_ridge = [{'max_iter':[5000],
        'alpha':[1e-10, 1e-8, 1e-4,1e-2, 1, 5, 10, 100],
        'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        }]  
        ridge_random = GridSearchCV(Ridge(),refit=True, param_grid = param_grid_ridge,iid=True, cv=5)
        ridge_random.fit(X_train,y_train)
        print('------------------------------------------')
        print('tuning Ridge')
        print('------------------------------------------')
        print(ridge_random.best_params_)
        print(ridge_random.best_score_)
        print(rmsle(ridge_random.predict(X_train),y_train))

        return ridge_random.best_estimator_

    def GradientBoost_autotune(self, X_train, y_train):
        param_grid_GradientBoost = [{'max_depth':[3,5],
            'learning_rate':[0.1, 0.01, 0.3],
            'alpha':[0.5, 0.1, 0.9]}] 
        GradientBoost_random = GridSearchCV(Ridge(),refit=True, param_grid = param_grid_GradientBoost,iid=False, cv=5)
        GradientBoost_random.fit(X_train,y_train)
        print('------------------------------------------')
        print('tuning GradientBoost')
        print('------------------------------------------')
        print(GradientBoost_random.best_params_)
        print(GradientBoost_random.best_score_)
        print(rmsle(GradientBoost_random.predict(X_train),y_train))

        return GradientBoost_random.best_estimator_        

# Run
BDP = bicycle_Demand_project()
train_data = BDP.datasetLoad("train")
test_data = BDP.datasetLoad("test")
train_data = BDP.datasetPreprocessing(train_data,"preprocessed_train_2021-12-14", is_it_train=True)
test_data = BDP.datasetPreprocessing(test_data,"preprocessed_test_2021-12-14", is_it_train=False)
X_train, X_valid, y_train, y_valid = BDP.data_split(train_data)

##### 여기서부터 나는 에러 해결해야함 
## 점수내는부분에서 문제인건지 ... 
## fit 할때 에러니까 기존 내가 짜뒀던 방식이랑 베낀 코드랑 비교해봐야할듯 
# https://www.kaggle.com/carolineecc/xgboost-random-forest-ridge-lasso-regression 


%%time
ridge_param = BDP.ridge_autotune(X_train,y_train)
gb_param = BDP.GradientBoost_autotune(X_train,y_train)
random_forest_param = BDP.randomforest_autotunes(X_train,y_train)

# 추후 처리 
    # ## 만약 train 데이터라면 count를 log 처리 해줌 + test 데이터에 없는 컬럼 제거
    # if is_it_train :
    #     dataset['log_count'] = np.log1p(dataset['count'])
    #     dataset.drop(['count','casual','registered'], axis = 1, inplace = True)



# def fit_model_step(self,trainX,trainy,valX,valy):
#     start_time = time.time()
#     model = 
#     model.fit()
#     print(time.time() - start_timt,'sec') 

# # 학습시간 카운트        
# return model 

# def fit_models(self,X_train_val, y_train_val):
#     X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,test_size = 0.25)        
#     self.fit_model_step(X_train, y_train, X_val, y_val)
#     model.save()
    
# def modelTrain(self,dataset):
#     start_time = time.time()
#     features, target = dataset.drop(self.y,axis =1), dataset[self.y]_scater = self.scaler     
#     self.fit_models(features, target)        
#     print(f'>>>Processing time : {time.time() - start_time}sec--'')
    
# def getScaler(self, dataset):
# self.scaler = 
    
# def testsetPreprocessing(self, datset):

    
# def trainingsetPredict(self, dataset):
        
# _pred = 
# self.ensemble_predictions(X_test)

#     return testset_result


    
# def load_models(self, models):        
# _model_list = [ ]       
# for i in models:
#      _moel_list.append()
       
#     return model_list 
    
# def ensemble_predictions(self,testX):       
#     models = 
#     self.load_models()    
#     predictions = _model.predict()    
#     # ~~~ ensemble 
#     return predictions