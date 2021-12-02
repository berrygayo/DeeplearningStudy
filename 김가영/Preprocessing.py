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
path = 'C:/Users/Berry/Documents/GitHub/DeeplearningStudy/김가영/'
os.chdir(path)

class bicycle_Demand_project:   
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.dataset_preprocessed = None
        self.loaded_data = None
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

    def datasetPreprocessing(self, dataset, saving_name):
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
    
        ## 불필요한 기존 year, temp, season, dayofweek, hour, month, weather 컬럼 drop
        dataset.drop(['temp', 'year', 'season', 'dayofweek', 'hour', 'month', 'weather'], axis = 1, inplace = True)
        
        self.dataset_preprocessed = dataset

        print(">>> Complete Preprocessing <<< ")  

        # save preprocessed_csv
        self.dataset_preprocessed.to_csv(f'{saving_name}.csv')
        print(">>> Success saving Preprocessing csv <<< ")   

        return self.dataset_preprocessed

# Run
BDP = bicycle_Demand_project()
train_data = BDP.datasetLoad("train")
test_data = BDP.datasetLoad("test")
BDP.datasetPreprocessing(train_data,"preprocessed_train_2021-12-02")
BDP.datasetPreprocessing(test_data,"preprocessed_test_2021-12-02")


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