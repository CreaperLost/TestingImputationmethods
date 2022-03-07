from numpy.core.numeric import NaN
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import KFold
import time
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.metrics import accuracy_score,roc_auc_score
import sklearn
from dae import DAE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder,LabelBinarizer
import glob
import datetime



def loop(dataset,sep=',',na_values='?',outcome_Type='binaryClass',problem='C',vmaps={},parameter={}):
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




        LL_train = np.transpose(X_train.values).tolist()
        LL_test  = np.transpose(X_test.values).tolist()

        start = time.time()
        imputer = DAE(parameters=parameter,names=column_names,vmaps=vmaps)


        Methods_Impute = imputer.fit(LL_train)
        

        #Transform returns List of List, New_Columns , New_Vmaps 
        Imputed_Train,Train_Column_names,Train_VMaps=Methods_Impute.transform(LL_train)
        Imputed_Test,Test_Column_names,Test_VMaps= Methods_Impute.transform(LL_test)



        total = total + time.time()-start
        #Turn LL to NP ARRAY
        Imputed_Train = np.transpose(np.array(Imputed_Train))
        Imputed_Test  = np.transpose(np.array(Imputed_Test))
        print(Imputed_Test.shape,Test_Column_names)        
        #NP ARRAY TO DF
        X__train_imputed = pd.DataFrame(Imputed_Train,columns=Train_Column_names) 
        X__test_imputed = pd.DataFrame(Imputed_Test,columns=Test_Column_names) 

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







#for file_name in glob.glob('realdata/'+'*.csv'): ,
for file_name in ['realdata\\pbcseq2.csv']: 
        
    if file_name == 'realdata\colleges_aaup.csv':
        categorical_features = ["State", "Type"]
        continue
    elif file_name == 'realdata\colleges_usnews.csv':
        categorical_features = ["State"]
        continue
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

    for ep in [500,1000]:
        for theta in [7,10]:
            for drop in [0.35,0.5]:
                error,total=loop(dataset=file_name,sep=';',na_values='?',outcome_Type='binaryClass',problem='C',vmaps=vmaps,parameter={"epochs":ep,"dropout":drop,"theta":theta,"Binary_Indicator":True})
                print('iter')
                print('Error for ' + file_name+   ' : ' + str(error) + '  Time : ' + str(datetime.timedelta(seconds=total)) + ' Params  : ' + str({"epochs":ep,"dropout":drop,"theta":theta}) )
                with open('dae+bi.txt', 'a') as the_file:
                    the_file.write('Error for ' + file_name+   ' : ' + str(error) + '  Time : ' + str(datetime.timedelta(seconds=total)) + ' Params  : ' + str({"epochs":ep,"dropout":drop,"theta":theta})+  '\n' )