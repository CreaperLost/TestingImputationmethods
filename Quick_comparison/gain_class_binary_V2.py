# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# Necessary packages
import tensorflow as tf
##IF USING TF 2 use following import to still use TF < 2.0 Functionalities
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
# Packages
import numpy as np
from tqdm.notebook import tqdm_notebook as tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import pandas as pd
import time
import seaborn as sns
sns.set()
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np
import glob
import datetime
import numpy as np
from tqdm import tqdm
from utils import normalization, renormalization, rounding
from utils import xavier_init
from utils import binary_sampler, uniform_sampler, sample_batch_index
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import MissingIndicator
import pickle
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils
import torch
import numpy as np
import torch.nn.functional as F
"""
def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model

# Hotfix function
def make_keras_picklable():

    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__

# Run the function
make_keras_picklable()
"""
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time



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




    
    kf = KFold(n_splits=2,shuffle=True,random_state=100)
    for train_index, test_index in kf.split(x):
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y[test_index]

        LL_train = np.transpose(X_train.values).tolist()
        LL_test  = np.transpose(X_test.values).tolist()

        column_names = list(X_train.columns)


        start = time.time()
        imputer = Gain(parameters=parameter,vmaps=vmaps,names=column_names)


        Methods_Impute = imputer.fit(LL_train)


        with open('gain.obj', 'wb') as f:
            pickle.dump(Methods_Impute, f)

        with open('gain.obj', 'rb') as f:
            imputer = pickle.load(f)
        #Transform returns List of List, New_Columns , New_Vmaps 
        Imputed_Train,Train_Column_names,Train_VMaps=Methods_Impute.transform(LL_train)
        Imputed_Test,Test_Column_names,Test_VMaps= Methods_Impute.transform(LL_test)

        total = total + time.time()-start

        #Turn LL to NP ARRAY
        Imputed_Train = np.transpose(np.array(Imputed_Train))
        Imputed_Test  = np.transpose(np.array(Imputed_Test))
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
    return error/2,total


class Gain():
    '''Impute missing values in data_x
    
    Args:
        - data_x: original data with missing values
        - gain_parameters: GAIN network parameters:
        - batch_size: Batch size
        - hint_rate: Hint rate
        - alpha: Hyperparameter
        - iterations: Iterations
        
    Returns:
        - imputed_data: imputed data
    '''
    def __init__(self,parameters: dict, names: list, vmaps: dict) -> None:
        # System parameters
        self.batch_size = parameters.get('batch_size',64)
        self.hint_rate = parameters.get('hint_rate',0.9)
        self.alpha = parameters.get('alpha',1)
        self.iterations = parameters.get('iterations',10000)
        self.Indi =  parameters.get('Binary_Indicator',False)

        self.theta_G= None
        self.names = names
        self.new_names = names
        self.vmaps = vmaps
        self.new_vmaps = vmaps
        torch.set_num_threads(6)

        #The indexes for categorical features in feature list.
        self.catindx = [names.index(i) for i in vmaps.keys()]
        self.numindx = [names.index(i) for i in names if i not in vmaps.keys()]
        self.cat_names = [i for i in vmaps.keys()]
        self.num_names = [i for i in names if i not in vmaps.keys()]
  
    # 1. Generator
    def generator(self,new_x,m):
        inputs = torch.cat(dim = 1, tensors = [new_x,m])  # Mask + Data Concatenate
        G_h1 = F.relu(torch.matmul(inputs, self.theta_G[0]) + self.theta_G[3])
        G_h2 = F.relu(torch.matmul(G_h1, self.theta_G[1]) + self.theta_G[4])   
        G_prob = torch.sigmoid(torch.matmul(G_h2, self.theta_G[2]) + self.theta_G[5]) # [0,1] normalized Output
                
        return G_prob

    def fit(self,X):
        #Code by George Paterakis
        #List of Lists -->> np.array with samples as rows and features as columns.
        data_x=np.transpose(np.array(X))

        if self.Indi == True:
            self.Indicator_var = MissingIndicator(error_on_new=False)
            self.Indicator_var.fit(data_x)

        # Define mask matrix
        data_m = 1-np.isnan(data_x)
        
        # Other parameters
        no, dim = data_x.shape
        
        # Hidden state dimensions
        h_dim = int(dim)
        
        # Normalization
        norm_data, self.norm_parameters = normalization(data_x)
        norm_data_x = np.nan_to_num(norm_data, 0)
        

        #Discriminator
        D_W1 = torch.tensor(xavier_init([dim*2, h_dim]),requires_grad=True)     # Data + Hint as inputs
        D_b1 = torch.tensor(np.zeros(shape = [h_dim]),requires_grad=True)

        D_W2 = torch.tensor(xavier_init([h_dim, h_dim]),requires_grad=True)
        D_b2 = torch.tensor(np.zeros(shape = [h_dim]),requires_grad=True)

        D_W3 = torch.tensor(xavier_init([h_dim, h_dim]),requires_grad=True)
        D_b3 = torch.tensor(np.zeros(shape = [h_dim]),requires_grad=True)       # Output is multi-variate

        theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
        

        #Generator
        G_W1 = torch.tensor(xavier_init([dim*2, h_dim]),requires_grad=True)     # Data + Mask as inputs (Random Noises are in Missing Components)
        G_b1 = torch.tensor(np.zeros(shape = [h_dim]),requires_grad=True)

        G_W2 = torch.tensor(xavier_init([h_dim, h_dim]),requires_grad=True)
        G_b2 = torch.tensor(np.zeros(shape = [h_dim]),requires_grad=True)

        G_W3 = torch.tensor(xavier_init([h_dim, h_dim]),requires_grad=True)
        G_b3 = torch.tensor(np.zeros(shape = [h_dim]),requires_grad=True)

        self.theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]



        

        # 2. Discriminator
        def discriminator(new_x, h):
            inputs = torch.cat(dim = 1, tensors = [new_x,h])  # Hint + Data Concatenate
            D_h1 = F.relu(torch.matmul(inputs, D_W1) + D_b1)  
            D_h2 = F.relu(torch.matmul(D_h1, D_W2) + D_b2)
            D_logit = torch.matmul(D_h2, D_W3) + D_b3
            D_prob = torch.sigmoid(D_logit)  # [0,1] Probability Output
            
            return D_prob


        def discriminator_loss(M, New_X, H):
            # Generator
            G_sample = self.generator(New_X,M)
            # Combine with original data
            Hat_New_X = New_X * M + G_sample * (1-M)

            # Discriminator
            D_prob = discriminator(Hat_New_X, H)

            #%% Loss
            D_loss = -torch.mean(M * torch.log(D_prob + 1e-8) + (1-M) * torch.log(1. - D_prob + 1e-8))
            return D_loss

        def generator_loss(X, M, New_X, H):
            #%% Structure
            # Generator
            G_sample = self.generator(New_X,M)

            # Combine with original data
            Hat_New_X = New_X * M + G_sample * (1-M)

            # Discriminator
            D_prob = discriminator(Hat_New_X, H)

            #%% Loss
            G_loss1 = -torch.mean((1-M) * torch.log(D_prob + 1e-8))
            MSE_train_loss = torch.mean((M * New_X - M * G_sample)**2) / torch.mean(M)

            G_loss = G_loss1 + self.alpha * MSE_train_loss 

            #%% MSE Performance metric
            MSE_test_loss = torch.mean(((1-M) * X - (1-M)*G_sample)**2) / torch.mean(1-M)
            return G_loss, MSE_train_loss, MSE_test_loss

        

        optimizer_D = torch.optim.Adam(params=theta_D)
        optimizer_G = torch.optim.Adam(params=self.theta_G)

        
        
        #%% Start Iterations
        for it in tqdm(range(self.iterations)):    
            
            #%% Inputs
            mb_idx = sample_batch_index(no, min(self.batch_size,no))
            X_mb = norm_data_x[mb_idx,:]  
            
            Z_mb = uniform_sampler(0, 0.01,  min(self.batch_size,no), dim) 
            M_mb = data_m[mb_idx,:]  
            # Sample hint vectors
            H_mb_temp = binary_sampler(self.hint_rate, min(self.batch_size,no), dim)
            H_mb = M_mb * H_mb_temp
           
            
            New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  # Missing Data Introduce
            
            X_mb = torch.tensor(X_mb).double()
            M_mb = torch.tensor(M_mb).double()
            H_mb = torch.tensor(H_mb).double()
            New_X_mb = torch.tensor(New_X_mb).double()
            
            optimizer_D.zero_grad()
            D_loss_curr = discriminator_loss(M=M_mb, New_X=New_X_mb, H=H_mb)
            D_loss_curr.backward()
            optimizer_D.step()
            
            optimizer_G.zero_grad()
            G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = generator_loss(X=X_mb, M=M_mb, New_X=New_X_mb, H=H_mb)
            G_loss_curr.backward()
            optimizer_G.step()    
                
            #%% Intermediate Losses
            if it % 1000 == 0:
                print('Iter: {}'.format(it),end='\t')
                print('Train_loss: {:.4}'.format(np.sqrt(MSE_train_loss_curr.item())),end='\t')
                print('Test_loss: {:.4}'.format(np.sqrt(MSE_test_loss_curr.item())))
        return self
               
    def transform(self,X):
        #Code by George Paterakis
        #List of Lists -->> np.array with samples as rows and features as columns.
        data_x=np.transpose(np.array(X))

        if self.Indi==True:
            X_indi = self.Indicator_var.transform(data_x)
            extra_col = ['Bi_' + str(i) for i in range(X_indi.shape[1])]
        # Define mask matrix
        data_m = 1-np.isnan(data_x)
    
        # Other parameters
        no, dim = data_x.shape
    
        # Hidden state dimensions
        h_dim = int(dim)
    
        # Normalization
        norm_data, norm_parameters = normalization(data_x,self.norm_parameters)
        norm_data_x = np.nan_to_num(norm_data, 0)

        ## Return imputed data      
        Z_mb = uniform_sampler(0, 0.01, no, dim) 
        M_mb = data_m
        X_mb = norm_data_x          
                
        New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  # Missing Data Introduce

        X_mb = torch.tensor(X_mb).double()
        M_mb = torch.tensor(M_mb).double()
        New_X_mb = torch.tensor(New_X_mb).double()
        
        def test_loss(X, M, New_X):
            #%% Structure
            # Generator
            G_sample = self.generator(New_X,M)

            #%% MSE Performance metric
            MSE_test_loss = torch.mean(((1-M) * X - (1-M)*G_sample)**2) / torch.mean(1-M)
            return MSE_test_loss, G_sample

        MSE_final, Sample = test_loss(X=X_mb, M=M_mb, New_X=New_X_mb)
                
        #print('Final Test RMSE: ' + str(np.sqrt(MSE_final.item())))





        
        imputed_data = M_mb * X_mb + (1-M_mb) * Sample
 
        imputed_data =imputed_data.detach().numpy()
        # Renormalization
        imputed_data = renormalization(imputed_data, self.norm_parameters)  
        
        # Rounding
        #if categoricals
        if len(self.catindx) >0 :
            imputed_data = rounding(imputed_data, self.catindx)  

        if self.Indi == True:
            imputed_data  = np.concatenate((imputed_data ,X_indi),axis=1)
            new_names = self.new_names + extra_col
            X=np.transpose(np.array(imputed_data)).tolist()
            return X,new_names, self.new_vmaps

        imputed_data = np.transpose(np.array(imputed_data)).tolist()

        return imputed_data,self.names,self.vmaps




for file_name in glob.glob('realdata/'+'*.csv'):
#for file_name in ['realdata\\schizo.csv']: 
    if file_name == 'realdata\lymphoma_2classes.csv':
        continue
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
    else:
        categorical_features = []
    vmaps=dict(zip(categorical_features, ['' for i in categorical_features]))

    for ep in [5000]:
        for hi_p in [0.9]:
            for alp in [10]:
                error , total = loop(dataset=file_name,sep=';',na_values='?',outcome_Type='binaryClass',problem='C',vmaps=vmaps,parameter={"iterations":ep,"hint_rate":hi_p,"alpha":alp,"Binary_Indicator":True})
                print('Error for ' + file_name+   ' : ' + str(error) + '  Time : ' + str(datetime.timedelta(seconds=total)) + ' Params  : ' + str({"epochs":ep,"hint_rate":hi_p,"alpha":alp}) )
                with open('gain_res+bi.txt', 'a') as the_file:
                    the_file.write('Error for ' + file_name+   ' : ' + str(error) + '  Time : ' + str(datetime.timedelta(seconds=total)) + ' Params  : ' + str({"epochs":ep,"hint_rate":hi_p,"alpha":alp}) + '\n' )



  

  

df = pd.read_csv('realdata\lymphoma_2classes.csv',na_values='?',sep=';',header=None)
outcome_Type = 'binaryClass'

y = pd.Series(LabelEncoder().fit_transform(df[df.columns[-1]]))
x = df.drop(df.columns[-1],axis=1)


LL_train = np.transpose(x.values).tolist()
column_names = list(x.columns)


categorical_features = []
vmaps=dict(zip(categorical_features, ['' for i in categorical_features]))


        
imputer = Gain(parameters={"iterations":5,"hint_rate":0.5,"alpha":0.1,"Binary_Indicator":False},vmaps=vmaps,names=column_names)
Methods_Impute = imputer.fit(LL_train)
with open('gain.obj', 'wb') as f:
    pickle.dump(Methods_Impute, f)

with open('gain.obj', 'rb') as f:
    imputer = pickle.load(f)
Imputed_Train,Train_Column_names,Train_VMaps=Methods_Impute.transform(LL_train)