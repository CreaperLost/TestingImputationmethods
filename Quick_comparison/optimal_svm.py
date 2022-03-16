from h2o import show_progress
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






class Optimal_SVM_Imputation():

    def __init__(self,parameters: dict, names: list, vmaps: dict) -> None:
        # System parameters
        self.gamma = parameters.get('gamma','auto')

        self.names = names
        self.new_names = names
        self.vmaps = vmaps
        self.new_vmaps = vmaps

        self.model =  None
        #The indexes for categorical features in feature list.
        self.catindx = [names.index(i) for i in vmaps.keys()]
        self.numindx = [names.index(i) for i in names if i not in vmaps.keys()]
        self.cat_names = [i for i in vmaps.keys()]
        self.num_names = [i for i in names if i not in vmaps.keys()]
  

    def fit(self,X):
        #Code by George Paterakis
        #List of Lists -->> np.array with samples as rows and features as columns.
        missing_data=np.transpose(np.array(X))

        na_data=pd.DataFrame(missing_data,columns=self.names)

        col_cat = self.catindx
        if len(col_cat) > 0:
            na_data.iloc[:,col_cat] = na_data.iloc[:,col_cat].astype('category')
        
        self.method = iai.OptSVMImputationLearner(svm_gamma=self.gamma,treat_unknown_level_missing=True,show_progress=False,random_seed=1).fit(na_data)

        return self
               
    def transform(self,X):
        #Code by George Paterakis
        #List of Lists -->> np.array with samples as rows and features as columns.
        data_x=np.transpose(np.array(X))

        na_data=pd.DataFrame(data_x,columns=self.names)

        col_cat = self.catindx
        if len(col_cat) > 0:
            na_data.iloc[:,col_cat] = na_data.iloc[:,col_cat].astype('category')

        imputed_data=self.method.transform(na_data)

        imputed_data = np.transpose(np.array(imputed_data)).tolist()

        return imputed_data,self.names,self.vmaps



def loop(dataset,sep=',',na_values='?',outcome_Type='binaryClass',problem='C',vmaps={},parameter={}):

    if dataset == 'realdata\lymphoma_2classes.csv':
        return
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
        
        imputer = Optimal_SVM_Imputation(parameters=parameter,vmaps=vmaps,names=column_names)
        
        import pickle
        
        
        """
        col_cat = [i for i in vmaps.keys()]
        if len(col_cat) > 0:
            X_train[col_cat] = X_train[col_cat].astype('category')
            X_test[col_cat] = X_test[col_cat].astype('category')
        """

        LL_train = np.transpose(X_train.values).tolist()
        LL_test  = np.transpose(X_test.values).tolist()

        Methods_Impute = imputer.fit(LL_train)
        
        #treat_unknown_level_missing = 'true'
        Imputed_Train,Train_Column_names,Train_VMaps=Methods_Impute.transform(LL_train)
        Imputed_Test,Test_Column_names,Test_VMaps= Methods_Impute.transform(LL_test)
        

        total = total + time.time()-start
         
        #Turn LL to NP ARRAY
        X__train_imputed = np.transpose(np.array(Imputed_Train))
        X__test_imputed  = np.transpose(np.array(Imputed_Test))

    
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
#for file_name in ['realdata\MAR_50_zoo.csv']:    
    if file_name == 'realdata\colleges_aaup.csv':
        continue
        categorical_features = ["Type"]
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
    elif file_name == 'realdata\lymphoma_2classes.csv':
        continue
    elif file_name == 'realdata\MAR_50_zoo.csv':
        categorical_features = ['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','tail','domestic','catsize']
    else:
        categorical_features = []
    vmaps=dict(zip(categorical_features, ['' for i in categorical_features]))

    error,total=loop(dataset=file_name,sep=';',na_values='?',outcome_Type='binaryClass',problem='C',vmaps=vmaps,parameter={"C":'auto'})
    print('Error for ' + file_name+   ' : ' + str(error) + '  Time : ' + str(datetime.timedelta(seconds=total)) )
    with open('simpleimpute.txt', 'a') as the_file:
        the_file.write('Error for ' + file_name+   ' : ' + str(error) + '  Time : ' + str(datetime.timedelta(seconds=total)) +  '\n' )