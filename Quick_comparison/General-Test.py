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
from sklearn.compose import ColumnTransformer
from meanmode import mm


def dataset_cat_flag(file_name):

    categorical_features = []
    flag = 0

    if file_name == 'realdata/colleges_aaup.csv':
        categorical_features = ["State", "Type"]
    elif file_name == 'real_50/50-Train-jad_primary-tumor.csv':
        flag = 1
    elif file_name == 'realdata/colleges_usnews.csv':
        categorical_features = ["State"]
    elif file_name == 'realdata\heart-h.csv':
        categorical_features = ["sex","chest_pain","fbs","restecg","exang","slope","thal"]
    elif file_name == 'realdata\kdd_coil_1.csv':
        categorical_features = ["season","river_size","fluid_velocity"]
    elif file_name == 'realdata\meta.csv':
        categorical_features = ['DS_Name','Alg_Name']
    elif file_name == 'realdata/schizo.csv' or file_name=='real_50/50-Train-jad_schizo.csv':
        categorical_features = ['target','sex']
    elif file_name == 'realdata/pbcseq2.csv':
        categorical_features = ['status','drug','sex','presence_of_asictes','presence_of_hepatomegaly','presence_of_spiders']
    elif file_name == 'realdata\MAR_50_zoo.csv':
        categorical_features = ['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','tail','domestic','catsize']
    elif file_name == 'realdata\MAR_10_molecular-biology_promoters.csv':
        categorical_features = ['p-50', 'p-49', 'p-48', 'p-47', 'p-46', 'p-45', 'p-44', 'p-43', 'p-42', 'p-41', 'p-40', 'p-39', 'p-38', 'p-37', 'p-36', 'p-35', 'p-34', 'p-33', 'p-32', 'p-31', 'p-30', 'p-29', 'p-28', 'p-27', 'p-26', 'p-25', 'p-24', 'p-23', 'p-22', 'p-21', 'p-20', 'p-19', 'p-18', 'p-17', 'p-16', 'p-15', 'p-14', 'p-13', 'p-12', 'p-11', 'p-10', 'p-9', 'p-8', 'p-7', 'p-6', 'p-5', 'p-4', 'p-3', 'p-2', 'p-1', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7']
    elif file_name == 'realdata\MAR_50_churn.csv':
        categorical_features = ['international_plan','voice_mail_plan']
    elif file_name == 'realdata\MAR_50_boston.csv':
        categorical_features = ['CHAS']
    elif file_name == 'realdata\MAR_50_Australian.csv':
        categorical_features = ['A1','A8','A9','A11']
    elif file_name == 'realdata/jad_audiology.csv':
        categorical_features = ['age_gt_60', 'air', 'airBoneGap', 'ar_c', 'ar_u', 'bone', 'boneAbnormal', 'bser', 'history_buzzing', 'history_dizziness', 'history_fluctuating', 'history_fullness', 'history_heredity', 'history_nausea', 'history_noise', 'history_recruitment', 'history_ringing', 'history_roaring', 'history_vomiting', 'late_wave_poor', 'm_at_2k', 'm_cond_lt_1k', 'm_gt_1k', 'm_m_gt_2k', 'm_m_sn', 'm_m_sn_gt_1k', 'm_m_sn_gt_2k', 'm_m_sn_gt_500', 'm_p_sn_gt_2k', 'm_s_gt_500', 'm_s_sn', 'm_s_sn_gt_1k', 'm_s_sn_gt_2k', 'm_s_sn_gt_3k', 'm_s_sn_gt_4k', 'm_sn_2_3k', 'm_sn_gt_1k', 'm_sn_gt_2k', 'm_sn_gt_3k', 'm_sn_gt_4k', 'm_sn_gt_500', 'm_sn_gt_6k', 'm_sn_lt_1k', 'm_sn_lt_2k', 'm_sn_lt_3k', 'middle_wave_poor', 'mod_gt_4k', 'mod_mixed', 'mod_s_mixed', 'mod_s_sn_gt_500', 'mod_sn', 'mod_sn_gt_1k', 'mod_sn_gt_2k', 'mod_sn_gt_3k', 'mod_sn_gt_4k', 'mod_sn_gt_500', 'notch_4k', 'notch_at_4k', 'o_ar_c', 'o_ar_u', 's_sn_gt_1k', 's_sn_gt_2k', 's_sn_gt_4k', 'speech', 'static_normal', 'tymp', 'viith_nerve_signs', 'wave_V_delayed', 'waveform_ItoV_prolonged']
    elif file_name == 'realdata\MAR_50_churn.csv':
        categorical_features = ['international_plan','voice_mail_plan']
    else:
        categorical_features = []
    
    return (categorical_features,flag)


def loop(dataset,sep=',',na_values='?',outcome_Type='binaryClass',problem='C',vmaps={},parameter={},Model=None,flag=0,test_data = None,test_labels=None):
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
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]


        column_names = list(X_train.columns)

        if flag == 1:
            tr= OrdinalEncoder(unknown_value=np.nan,handle_unknown="use_encoded_value")
            X_train = pd.DataFrame(tr.fit_transform(X_train))
            X_test = pd.DataFrame(tr.transform(X_test))
            Imputed_Train = X_train.values
            Imputed_Test = X_test.values
            categorical_features = column_names
            vmaps=dict(zip(categorical_features, ['' for i in categorical_features]))
            #print('All data are categorical',categorical_features)
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
            
            Imputed_Train=sclr.fit_transform(X_train)
            Imputed_Test=sclr.transform(X_test)


        LL_train = np.transpose(Imputed_Train).tolist()
        LL_test  = np.transpose(Imputed_Test).tolist()

        start = time.time()
        imputer = Model(parameters=parameter,names=column_names,vmaps=vmaps)
        
        Methods_Impute = imputer.fit(LL_train)
        

        #Transform returns List of List, New_Columns , New_Vmaps 
        Imputed_Train,Train_Column_names,Train_VMaps=Methods_Impute.transform(LL_train)
        
        Imputed_Test,Test_Column_names,Test_VMaps= Methods_Impute.transform(LL_test)



        total = total + time.time()-start
        #Turn LL to NP ARRAY
        Imputed_Train = np.transpose(np.array(Imputed_Train))
        Imputed_Test  = np.transpose(np.array(Imputed_Test))
        #print(pd.DataFrame(Imputed_Train).head(5))
        
        #NP ARRAY TO DF
        X__train_imputed = pd.DataFrame(Imputed_Train,columns=column_names) 
        #print(X__train_imputed)
        X__test_imputed = pd.DataFrame(Imputed_Test,columns=column_names) 


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

    #Extra Step
    if test_data != None:
        column_names = list(x.columns)
        #Retrain on the whole dataset. Then Apply.
        if flag == 1:
            tr= OrdinalEncoder(unknown_value=np.nan,handle_unknown="use_encoded_value")
            x = pd.DataFrame(tr.fit_transform(x)).values
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
            x=sclr.fit_transform(x)
        
        imputer = Model(parameters=parameter,names=column_names,vmaps=vmaps)
        x = np.transpose(x).tolist()
        Methods_Impute = imputer.fit(x)
        #Transform returns List of List, New_Columns , New_Vmaps 

        Imputed_X,Train_Column_names,Train_VMaps=Methods_Impute.transform(x)
        Imputed_X  = np.transpose(np.array(Imputed_X))
        #print(pd.DataFrame(Imputed_Train).head(5))
        
        #NP ARRAY TO DF
        Imputed_X_ready = pd.DataFrame(Imputed_X,columns=column_names) 
        RF_Trained = RF_Model.fit(Imputed_X_ready,y)


        df = pd.read_csv(test_data,na_values=na_values,sep=sep)
        out = pd.read_csv(test_labels,header=None)
        if flag == 1:
            
            df = pd.DataFrame(tr.transform(df))
            holdout = df.values
        else:
            holdout=sclr.transform(df)
        LL_Holdout  = np.transpose(holdout).tolist()
        Imputed_Holdout,Test_Column_names,Test_VMaps= Methods_Impute.transform(LL_Holdout)
        Imputed_Holdout  = np.transpose(np.array(Imputed_Holdout))        
        Imputed_Holdout = pd.DataFrame(Imputed_Holdout,columns=column_names) 
        preds_holdout=RF_Trained.predict(Imputed_Holdout)
        #print(preds_holdout)
        print('Holdout AUC is : ',roc_auc_score(out,preds_holdout))
    return error/5,total



for file_name in [
    #('real_50/50-Train-jad_primary-tumor.csv','real_50/50-Test-Features-jad_primary-tumor.csv','real_50/50-Test-Label-jad_primary-tumor.csv'),
    #('real_50/50-Train-jad_water-treatment.csv','real_50/50-Test-Features-jad_water-treatment.csv','real_50/50-Test-Label-jad_water-treatment.csv'),
    ('real_50/50-Train-jad_schizo.csv','real_50/50-Test-Features-jad_schizo.csv','real_50/50-Test-Label-jad_schizo.csv')

    ]:
#for file_name in ['realdata2/50-Train-jad_primary-tumor.csv']:
#for file_name in ['realdata/jad_primary-tumor.csv']:
#for file_name in glob.glob('realdata/'+'*.csv'):
    f_name = file_name[0]
    t_feat= file_name[1]
    t_label = file_name[2]
    categorical_features,flag=dataset_cat_flag(f_name)

    print('Cat Features : , flag',categorical_features,flag)    
    vmaps=dict(zip(categorical_features, ['' for i in categorical_features]))
    
    #error,total=loop(dataset=f_name,sep=',',na_values='?',outcome_Type='binaryClass',problem='C',vmaps=vmaps,parameter={"epochs":100,"dropout":0.5,"theta":7,"lr":0.01},Model=DAE,flag=flag,test_data=t_feat,test_labels=t_label)
    #print('DAE Error for ' + f_name+   ' : ' + str(error) + '  Time : ' + str(datetime.timedelta(seconds=total)) + ' Params  : ' + str({"epochs":100,"dropout":0.5,"theta":7,"lr":0.01}) )
    
    error,total=loop(dataset=f_name,sep=',',na_values='?',outcome_Type='binaryClass',problem='C',vmaps=vmaps,parameter={"epochs":100,"dropout":0.5,"theta":7,"lr":0.01},Model=mm,flag=flag,test_data=t_feat,test_labels=t_label)
    print('MeanMode Error for ' + f_name+   ' : ' + str(error) + '  Time : ' + str(datetime.timedelta(seconds=total)) + ' Params  : ' + str("MM") )
