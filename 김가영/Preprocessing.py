# Randomstate 통일 과정 필요 : kf = KFold(n_splits = n, shuffle=True, random_state = ~~ )
# 지혜님 preprocessing > 변수 두개 추가


# Package loading


import numpy
as np 


class bicycle_Demand_project:   
    def__init__(self):
        self.train_data = None
    self test_data = none 
    self.continuous_col = [' ~~연속형 변수 ~~~ ']
    self.categorical_col = ['~~~ 범주형 변수 ~~~ ']   
    self.binary_col = ['~~~ 이진 변수 ~~~']
    self.y = ['counts']
    self.scaler = {}       
    self.feature_column = {}       
    self.dataset_preprocessed = None


def datasetLoad(self):
    path = '/~~~ 경로 '
    self.train_data = pd.read_csv()
    self.test_data = pd.read_csv()

def datasetPreprocessing(self,dataset):
    self.dataset_preprocessed = 
    deffit_model_step(self,trainX,trainy,valX,valy):
    start_time = time.time()
    model = 
    model.fit()
    print(time.time() - start_timt,'sec') 

# 학습시간 카운트        
return model 

def fit_models(self,X_train_val, y_train_val):
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,test_size = 0.25)        
    self.fit_model_step(X_train, y_train, X_val, y_val)
    model.save()
    
def modelTrain(self,dataset):
    start_time = time.time()
    features, target = dataset.drop(self.y,axis =1), dataset[self.y]_scater = self.scaler     
    self.fit_models(features, target)        
    print(f'>>>Processing time : {time.time() - start_time}sec--'')
    
def getScaler(self, dataset):
self.scaler = 
    
def testsetPreprocessing(self, datset):

    
def trainingsetPredict(self, dataset):
        
_pred = 
self.ensemble_predictions(X_test)

    return testset_result


    
def load_models(self, models):        
_model_list = [ ]       
for i in models:
     _moel_list.append()
       
    return model_list 
    
def ensemble_predictions(self,testX):       
    models = 
    self.load_models()    
    predictions = _model.predict()    
    # ~~~ ensemble 
    return predictions