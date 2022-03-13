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
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder,LabelBinarizer
from sklearn.impute import SimpleImputer,MissingIndicator
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
        
        imputer = ColumnTransformer(
            transformers=[
                ("cat", SimpleImputer(strategy='most_frequent'), [column_names.index(i) for i in vmaps.keys()]),
                ("num", SimpleImputer(strategy='mean'), [column_names.index(i) for i in column_names if i not in vmaps.keys()]),
            ],remainder='passthrough'
        )


        Indi = MissingIndicator(error_on_new=False)


        #print([column_names.index(i) for i in vmaps.keys()],[column_names.index(i) for i in column_names if i not in vmaps.keys()])
        #imputer = SimpleImputer()
        Methods_Impute = imputer.fit(X_train.values)
        Indi.fit(X_train.values)

        Imputed_Train=Methods_Impute.transform(X_train.values)
        Bi_train = Indi.transform(X_train.values)



        Imputed_Test= Methods_Impute.transform(X_test.values)
        Bi_test = Indi.transform(X_test.values)


        total = total + time.time()-start
         
        #NP ARRAY TO DF
        X__train_imputed = pd.DataFrame(Imputed_Train,columns=column_names) 
        X__test_imputed = pd.DataFrame(Imputed_Test,columns=column_names) 


        X__train_imputed = pd.concat([X__train_imputed,pd.DataFrame(Bi_train,columns= ['Bi_' + str(i) for i in range(Bi_train.shape[1])])],axis=1)
        X__test_imputed =  pd.concat([X__test_imputed,pd.DataFrame(Bi_test,columns =['Bi_' + str(i) for i in range(Bi_train.shape[1])])],axis=1)

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


       
for file_name in glob.glob('realdata/'+'*.csv'):
    
    if file_name == 'realdata\colleges_aaup.csv':
        categorical_features = ["State", "Type"]
    elif file_name == 'realdata\colleges_usnews.csv':
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
    else:
        categorical_features = []
    vmaps=dict(zip(categorical_features, ['' for i in categorical_features]))


    error,total=loop(dataset=file_name,sep=';',na_values='?',outcome_Type='binaryClass',problem='C',vmaps=vmaps)
    print('Error for ' + file_name+   ' : ' + str(error) + '  Time : ' + str(datetime.timedelta(seconds=total)) )
    with open('BI+MEAN.txt', 'a') as the_file:
        the_file.write('Error for ' + file_name+   ' : ' + str(error) + '  Time : ' + str(datetime.timedelta(seconds=total)) +  '\n' )