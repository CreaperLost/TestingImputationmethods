from configparser import NoSectionError
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
from new_dae import DAE
#from dae_mix_encoding import DAE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder,LabelBinarizer
import glob
import datetime
from sklearn.compose import ColumnTransformer
from meanmode import mm
from gain import Gain
from missForest import IterativeImputer
from ppca_code import PPCA

pd.set_option('display.max_columns', None)

def dataset_cat_flag(file_name):

    categorical_features = []
    flag = 0
    
    if file_name == 'real_50\\50-Train-jad_analcatdata_reviewer.csv':
        flag = 1
    elif file_name == 'real_50\\50-Train-jad_anneal.csv':
        categorical_features = ['family','product-type','steel','temper_rolling','condition','non-ageing','surface-finish','surface-quality','bc','bf','bt','bw/me','bl','chrom','phos','cbond','exptl','ferro','blue/bright/varn/clean','lustre','shape','oil']
    elif file_name == 'real_50\\50-Train-jad_audiology.csv':
        flag = 1
    elif file_name == 'real_50\\50-Train-jad_autoHorse.csv':
        categorical_features= ['fuel-type', 'aspiration', 'body-style', 'drive-wheels', 'make','engine-location', 'engine-type',  'fuel-system']
    elif file_name == 'real_50\\50-Train-jad_bridges.csv':
        categorical_features= ['RIVER', 'PURPOSE', 'CLEAR-G', 'T-OR-D', 'MATERIAL', 'SPAN', 'REL-L']
    elif file_name == 'real_50\\50-Train-jad_cjs.csv':
        categorical_features= [ 'TREE', 'BR']
    elif file_name == 'real_50\\50-Train-jad_colic.csv':
        categorical_features = ['surgery', 'Age', 'temp_extremities', 'peripheral_pulse', 'mucous_membranes', 'capillary_refill_time', 
        'pain', 'peristalsis', 'abdominal_distension', 'nasogastric_tube', 'nasogastric_reflux', 
        'rectal_examination', 'abdomen', 'abdominocentesis_appearance','outcome']
    elif file_name == 'real_50\\50-Train-jad_colleges_aaup.csv':
        categorical_features = [ 'State', 'Type' ]
    elif file_name == 'real_50\\50-Train-jad_cylinder-bands.csv':
        categorical_features = [
        'cylinder_number', 'customer', 'grain_screened', 'ink_color', 'proof_on_ctd_ink', 
        'blade_mfg', 'cylinder_division', 'paper_type', 'ink_type', 'direct_steam', 'solvent_type', 
        'type_on_cylinder', 'press_type', 'cylinder_size', 'paper_mill_location']
    elif file_name == 'real_50\\50-Train-jad_dresses-sales.csv':
        categorical_features = ['V2', 'V3', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13']
    elif file_name == 'real_50\\50-Train-jad_eucalyptus.csv':
        categorical_features = ['Abbrev', 'Locality', 'Map_Ref', 'Latitude', 'Sp']
    elif file_name == 'real_50\\50-Train-jad_hepatitis.csv':
        categorical_features = ['SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE', 'ANOREXIA', 
        'LIVER_BIG', 'LIVER_FIRM', 'SPLEEN_PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES', 'HISTOLOGY']
    elif file_name == 'real_50\\50-Train-jad_mushroom.csv':
        flag = 1
    elif file_name == 'real_50\\50-Train-jad_pbcseq.csv':
        categorical_features = ['drug', 'sex', 'presence_of_asictes', 'presence_of_hepatomegaly', 
        'presence_of_spiders']
    elif file_name == 'real_50\\50-Train-jad_primary-tumor.csv':
        flag = 1
    elif file_name == 'real_50\\50-Train-jad_profb.csv':
        categorical_features = ['Favorite_Name', 'Underdog_name' ,'Weekday', 'Overtime']
    elif file_name == 'real_50\\50-Train-jad_schizo.csv':
        categorical_features = ['target', 'sex' ]
    elif file_name == 'real_50\\50-Train-jad_sick.csv':
        categorical_features = ['sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured','referral_source']
    elif file_name == 'real_50\\50-Train-jad_soybean.csv':
        flag = 1
    elif file_name == 'real_50\\50-Train-jad_stress.csv':
        categorical_features = ['Sexe', 'Consommation_tabac', 'type_consommation', 'Allergies']
    elif file_name == 'real_50\\50-Train-jad_vote.csv':
        flag = 1
    elif file_name == 'real_50\\50-Train-jad_hungarian.csv':
        categorical_features = ['sex' ,'fbs','exang']
    elif file_name == 'real_50\\50-Train-jad_braziltourism.csv':
        categorical_features = ['Sex', 'Access_road']
    else:
        categorical_features = []
    
    return (categorical_features,flag)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.impute import MissingIndicator

from softImpute import SoftImpute

class Indicator():
    
    def __init__(self,parameters: dict, names: list, vmaps: dict,
                 missing_values=np.nan):
       
        

        
        self.names = names
        self.new_names = names
        self.vmaps = vmaps
        self.new_vmaps = vmaps


        #The indexes for categorical features in feature list.
        self.catindx = sorted([names.index(i) for i in vmaps.keys()])
        self.numindx = sorted([names.index(i) for i in names if i not in vmaps.keys()])
        self.cat_names = [self.names[i] for i in self.catindx]
        self.num_names = [self.names[i] for i in self.numindx]
        self.missing_values = missing_values


    def _initial_imputation(self, X):
        
        X_filled = X.copy()
        indicator = MissingIndicator()
        indicator.fit(X_filled)
        X_filled = indicator.transform(X_filled)
        return X_filled



    def fit_transform(self, X, y=None):
        """Fits the imputer on X and return the transformed X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data, where "n_samples" is the number of samples and
            "n_features" is the number of features.

        y : ignored.

        Returns
        -------
        Xt : array-like, shape (n_samples, n_features)
            The imputed input data.
        """
        #Code by George Paterakis
        #List of Lists -->> np.array with samples as rows and features as columns.
        X=np.transpose(np.array(X))
        X_r = self._initial_imputation(X)
        X=np.transpose(np.array(X_r)).tolist()


        return X

    def fit(self, X, y=None):
        """Fits the imputer on X and return self.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data, where "n_samples" is the number of samples and
            "n_features" is the number of features.

        y : ignored

        Returns
        -------
        self : object
            Returns self.
        """

        self.fit_transform(X)
        return self





def loop(dataset,sep=',',na_values='?',outcome_Type='binaryClass',problem='C',vmaps={},parameter={},Model=None,flag=0,test_data = None,test_labels=None):

    df = pd.read_csv(dataset,na_values=na_values,sep=sep)
    outcome_Type = 'binaryClass'
    y = df[outcome_Type]
    x = df.drop(outcome_Type,axis=1)
    x = x.dropna(axis=1,how='all')
    column_names = list(x.columns)


    if flag == 1:
        tr= OrdinalEncoder(unknown_value=np.nan,handle_unknown="use_encoded_value")
        X_train = pd.DataFrame(tr.fit_transform(x))
        Imputed_Train = X_train.values
        categorical_features = column_names
        vmaps=dict(zip(categorical_features, ['' for i in categorical_features]))
    else:
        sclr = ColumnTransformer(
                transformers=[
                    ("std", 
                    StandardScaler(), 
                    [column_names.index(i) for i in column_names if i not in vmaps.keys()]),
                    ("ordi",
                    OrdinalEncoder(unknown_value=np.nan,handle_unknown="use_encoded_value"),
                    [column_names.index(i) for i in column_names if i in vmaps.keys()])
                    ],
                    remainder = 'passthrough'
            )
        column_names2 = [i for i in column_names if i not in vmaps.keys()] + [i for i in column_names if i in vmaps.keys()] 
        column_names = column_names2
        Imputed_Train=sclr.fit_transform(x)

    start = time.time()
    if  Model == SoftImpute:
        parameter = {"nPcs":len(column_names)-1,"lambda":0}
    elif Model == PPCA:
        parameter = {"nPcs":len(column_names)-1}

    imputer = Model(parameters=parameter,names=column_names,vmaps=vmaps)
    LL_train = np.transpose(Imputed_Train).tolist()
    Methods_Impute = imputer.fit(LL_train)
    fit_time = time.time()-start
    return fit_time
    
columns = ['Dataset','Mean','Gain','Miss','Soft','PPCA','DAE','Mean+BI','Gain+BI','Miss+BI','Soft+BI','PPCA+BI','DAE+BI']
Res = list()
for file_name in glob.glob('real_50/'+'*.csv'):
    
    f_name = file_name
    categorical_features,flag=dataset_cat_flag(f_name)

    print(f_name,'Cat Features : , flag',categorical_features,flag)    
    vmaps=dict(zip(categorical_features, ['' for i in categorical_features]))
    
    time_soft=loop(dataset=f_name,sep=',',na_values='?',outcome_Type='binaryClass',problem='C',vmaps=vmaps,parameter={"nPcs":5,"lambda":0},Model=SoftImpute,flag=flag,test_data=None,test_labels=None)
    print('Soft  Time : ' + str(time_soft) + ' Params  : ' + str("Soft") )

    time_ppca=loop(dataset=f_name,sep=',',na_values='?',outcome_Type='binaryClass',problem='C',vmaps=vmaps,parameter={"nPcs":5},Model=PPCA,flag=flag,test_data=None,test_labels=None)
    print('PPCA  Time : ' + str(time_ppca) + ' Params  : ' + str("PPCA") )

    time_dae=loop(dataset=f_name,sep=',',na_values='?',outcome_Type='binaryClass',problem='C',vmaps=vmaps,parameter={"epochs":500,"dropout":0.5,"theta":10,"lr":0.01},Model=DAE,flag=flag,test_data=None,test_labels=None)
    print('DAE  Time : ' + str(time_dae) + ' Params  : ' + str("DAE") )    

    time_mm=loop(dataset=f_name,sep=',',na_values='?',outcome_Type='binaryClass',problem='C',vmaps=vmaps,parameter={},Model=mm,flag=flag,test_data=None,test_labels=None)
    print('MeanMode  Time : ' + str(time_mm) + ' Params  : ' + str("MM") )

    time_Gain=loop(dataset=f_name,sep=',',na_values='?',outcome_Type='binaryClass',problem='C',vmaps=vmaps,parameter={"iterations":10000,"hint_rate":0.9,"alpha":0.1},Model=Gain,flag=flag,test_data=None,test_labels=None)
    print('GAIN  Time : ' + str(time_Gain) + ' Params  : ' + str("GAIN") )

    time_mf=loop(dataset=f_name,sep=',',na_values='?',outcome_Type='binaryClass',problem='C',vmaps=vmaps,parameter={"n_estimators":250,"max_depth":30},Model=IterativeImputer,flag=flag,test_data=None,test_labels=None)
    print('MF  Time : ' + str(time_mf) + ' Params  : ' + str("MF") )

    time_Indi=loop(dataset=f_name,sep=',',na_values='?',outcome_Type='binaryClass',problem='C',vmaps=vmaps,parameter={"n_estimators":250,"max_depth":30},Model=Indicator,flag=flag,test_data=None,test_labels=None)
    print('Indicator  Time : ' + str(time_Indi) + ' Params  : ' + str("Indicator") )
    Res.append([f_name,time_mm,time_Gain,time_mf,time_soft,time_ppca,time_dae,time_mm+time_Indi,time_Gain+time_Indi,time_mf+time_Indi,time_soft+time_Indi,time_ppca+time_Indi,time_dae+time_Indi])

pd.DataFrame(Res,columns = columns).to_csv('Time-Execution-Results.csv',header=columns,index=None,sep=';')
    