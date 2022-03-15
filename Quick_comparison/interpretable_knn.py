from interpretableai import iai
from numpy.core.numeric import NaN
import pandas as pd
import random
import numpy as np
from scipy.sparse import data
from sklearn.model_selection import KFold
import time
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder,LabelBinarizer
from sklearn.impute import SimpleImputer
import os
import datetime
import glob
import pandas as pd
from sklearn.compose import ColumnTransformer




def loop(dataset,sep=',',na_values='?',outcome_Type='binaryClass',problem='C',vmaps={}):


    if dataset == 'realdata\lymphoma_2classes.csv':
        df = pd.read_csv(dataset,na_values=na_values,sep=sep,header=None)
    else:
        df = pd.read_csv(dataset,na_values=na_values,sep=sep)

    

    if 'binaryClass' in df.columns:
        outcome_Type = 'binaryClass'
    else:
        outcome_Type = 'outcome'


    if dataset == 'realdata\lymphoma_2classes.csv':
        outcome_Type = 'binaryClass'
        y = pd.Series(LabelEncoder().fit_transform(df[df.columns[-1]]))
        x = df.drop(df.columns[-1],axis=1)
    else:
        y = df[outcome_Type]
        x = df.drop(outcome_Type,axis=1)
    
    total = 0
    error = 0

    
    kf = KFold(n_splits=5,shuffle=True,random_state=100)
    for train_index, test_index in kf.split(x):
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y[test_index]


        column_names = list(X_train.columns)

        start = time.time()
        
        imputer = iai.SingleKNNImputationLearner(knn_k = 5)
        
        col_cat = [i for i in vmaps.keys()]
        X_train[col_cat] = X_train[col_cat].astype('category')
        X_test[col_cat] = X_test[col_cat].astype('category')


        Methods_Impute = imputer.fit(X_train)

        Imputed_Train=Methods_Impute.transform(X_train)
        Imputed_Test= Methods_Impute.transform(X_test)
        
        print(Imputed_Train)
        print(Imputed_Test)

        total = total + time.time()-start
         
        #NP ARRAY TO DF
        #X__train_imputed = pd.DataFrame(Imputed_Train,columns=column_names) 
        #X__test_imputed = pd.DataFrame(Imputed_Test,columns=column_names) 
        X__train_imputed=Imputed_Train
        X__test_imputed = Imputed_Test
    
        if outcome_Type == 'binaryClass':
            RF_Model = RandomForestClassifier(random_state=0)
        else:
            RF_Model = RandomForestRegressor(random_state=0)
        
        RF_Trained = RF_Model.fit(X__train_imputed,y_train)
    
        preds=RF_Trained.predict(X__test_imputed)

        if outcome_Type == 'binaryClass':
            error += roc_auc_score(y_test,preds)
        else:
            error += r2_score(y_test,preds)
    
    return error/5,total


       
#for file_name in glob.glob('realdata/'+'*.csv'):
for file_name in ['realdata\MAR_50_zoo.csv']:    
    if file_name == 'realdata\colleges_aaup.csv':
        categorical_features = ["State", "Type"]
    elif file_name == 'realdata\colleges_usnews.csv':
        continue
        categorical_features = ["State"]
    elif file_name == 'realdata\heart-h.csv':
        categorical_features = ["sex","chest_pain","fbs","restecg","exang","slope","thal"]
    elif file_name == 'realdata\kdd_coil_1.csv':
        categorical_features = ["season","river_size","fluid_velocity"]
    elif file_name == 'realdata\meta.csv':
        categorical_features = ['DS_Name','Alg_Name']
    elif file_name == 'realdata\schizo.csv':
        categorical_features = ['target','sex']
    elif file_name == 'realdata\pbcseq2.csv':
        categorical_features = ['status','drug','sex','presence_of_asictes','presence_of_hepatomegaly','presence_of_spiders']
    elif file_name == 'realdata\MAR_50_zoo.csv':
        categorical_features = ['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','tail','domestic','catsize']
    else:
        categorical_features = []
    vmaps=dict(zip(categorical_features, ['' for i in categorical_features]))


    error,total=loop(dataset=file_name,sep=';',na_values='?',outcome_Type='binaryClass',problem='C',vmaps=vmaps)
    print('Error for ' + file_name+   ' : ' + str(error) + '  Time : ' + str(datetime.timedelta(seconds=total)) )
    with open('simpleimpute.txt', 'a') as the_file:
        the_file.write('Error for ' + file_name+   ' : ' + str(error) + '  Time : ' + str(datetime.timedelta(seconds=total)) +  '\n' )