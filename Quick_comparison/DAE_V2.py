from __future__ import division, print_function, absolute_import
from gain_v2_utils import initial_imputation, normalization, renormalization, onehot_decoding, onehot_encoding
from tqdm import tqdm
import tensorflow as tf
import numpy as np


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder,LabelBinarizer
import glob
import datetime
from sklearn.compose import ColumnTransformer




class DAE():
    
    def __init__(self,parameters: dict, names: list, vmaps: dict,
                 missing_values=np.nan):
       
        
        self.theta = parameters.get("theta",7)
        self.drop_out = parameters.get("dropout",0.5)
        self.batch_size = parameters.get("batch_size",64)
        self.epochs = parameters.get("epochs",500)
        self.lr = parameters.get("lr",0.01)
        self.dim = len(names)

        self.model = None
        
        
        torch.manual_seed(0)

        
        self.onehot  = OneHotEncoder(handle_unknown='ignore',sparse=False)

        
        self.names = names
        self.new_names = names
        self.vmaps = vmaps
        self.new_vmaps = vmaps


        #The indexes for categorical features in feature list.
        self.catindx = [names.index(i) for i in vmaps.keys()]
        self.numindx = [names.index(i) for i in names if i not in vmaps.keys()]
        self.cat_names = [i for i in vmaps.keys()]
        self.num_names = [i for i in names if i not in vmaps.keys()]
        self.missing_values = missing_values





    def _initial_imputation(self, X):
        
        

        X_filled = X.copy()

        #Code by George Paterakis
        #Mean Impute continous , Mode Impute Categorical


        #Mean Impute
        if len(self.numindx) >0 :
            if self.initial_imputer_Mean is None:
                self.initial_imputer_Mean = SimpleImputer(missing_values=self.missing_values,strategy='mean')
                X_filled[:,self.numindx] = self.initial_imputer_Mean.fit_transform(X[:,self.numindx])
            else:
                X_filled[:,self.numindx] = self.initial_imputer_Mean.transform(X[:,self.numindx])
           
        
        #Mode Impute
        if len(self.catindx) >0 :
            if self.initial_imputer_Mode is None:
                self.initial_imputer_Mode = SimpleImputer(missing_values=self.missing_values,strategy='most_frequent')
                X_filled[:,self.catindx] = self.initial_imputer_Mode.fit_transform(X[:,self.catindx])
            else:
                X_filled[:,self.catindx] = self.initial_imputer_Mode.transform(X[:,self.catindx])
            
        return X_filled


    @tf.function
    def encoder(self,x):

            x_noise = tf.cast(tf.nn.dropout(x, 0.5),dtype=tf.float32)
            layer_1 = tf.nn.tanh(tf.add(tf.matmul(x_noise, self.weights['encoder_h1']),
                                        self.biases['encoder_b1']))
            layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']),
                                        self.biases['encoder_b2']))
            layer_3 = tf.nn.tanh(tf.add(tf.matmul(layer_2, self.weights['encoder_h3']),
                                        self.biases['encoder_b3']))
            return layer_3

    # Building the decoder
    @tf.function
    def decoder(self,x):
            layer_1 = tf.nn.tanh(tf. add(tf.matmul( tf.cast(x,dtype=tf.float32), self.weights['decoder_h1']),
                                        self.biases['decoder_b1']))
            layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']),
                                        self.biases['decoder_b2']))
            layer_3 = tf.add(tf.matmul(layer_2, self.weights['decoder_h3']),
                            self.biases['decoder_b3'])
            col_index = 0
            empty_G_out = True
            # apply softmax to each categorical variable
            if self.catindx:
                empty_G_out = False
                output = tf.nn.softmax(layer_3[:, :self.n_classes[0]])
                col_index = self.n_classes[0]
                for j in range(1, len(self.n_classes)):
                    output = tf.concat(values=[output, tf.nn.softmax(layer_3[:, col_index:col_index + self.n_classes[j]])], axis=1)
                    col_index += self.n_classes[j]
            # apply sigmoid to all numerical variables
            if self.numindx:
                out_num = tf.nn.sigmoid(layer_3[:, col_index:])
                output = tf.concat(values=[output, out_num], axis=1) if not empty_G_out else out_num
            return output


    # sum up loss for each categorical variable
    @tf.function
    def dae_loss(self,y_pred, y_true, mask):
        loss = 0
        current_ind = 0
        # categorical loss
        if self.catindx:
            for j in range(len(self.n_classes)):
                mask_current = tf.cast(mask[:, current_ind:current_ind + self.n_classes[j]],dtype=tf.float32)
                y_pred_current = tf.cast(y_pred[:, current_ind:current_ind + self.n_classes[j]],dtype=tf.float32)
                y_true_current = tf.cast(y_true[:, current_ind:current_ind + self.n_classes[j]],dtype=tf.float32)

                loss += -tf.reduce_mean(input_tensor=mask_current * y_true_current * tf.cast(tf.math.log(mask_current * y_pred_current + 1e-8),dtype=tf.float32)) / tf.reduce_mean(input_tensor=mask_current)
                current_ind += self.n_classes[j]
        # numerical loss
        if self.numindx:
            mask_current = tf.cast(mask[:, current_ind:],dtype=tf.float32)
            y_pred_current = tf.cast(y_pred[:, current_ind:],dtype=tf.float32)
            y_true_current = tf.cast(y_true[:, current_ind:],dtype=tf.float32)
            loss += tf.reduce_mean((mask_current * y_true_current - mask_current * y_pred_current)**2) / tf.reduce_mean(mask_current)
        return loss

    # optimizer
    @tf.function
    def optimize_step(self,batch_x, batch_m):
        with tf.GradientTape() as g:
            y_hat = self.decoder(self.encoder(batch_x))
            l = self.dae_loss(y_hat, batch_x, batch_m)

        trainable_variables = list(self.weights.values()) + list(self.biases.values())

        gradients = g.gradient(l, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        return l, y_hat


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

        data_x=np.transpose(np.array(X))
        self.initial_imputer_Mean = None
        self.initial_imputer_Mode = None
        no, dim = data_x.shape
        # initial imputation
        data_x = self._initial_imputation(data_x)

        data_m = 1-np.isnan(data_x)
        self.all_levels = [np.unique(x)[~np.isnan(np.unique(x))] for x in data_x[:, self.catindx].T]


        data_train = np.array([])
        data_train_m = np.array([])
        ## encode cat
        if self.catindx:
            data_cat = data_x[:, self.catindx]
            data_cat_m = data_m[:, self.catindx]
            data_cat_enc, data_cat_enc_miss = onehot_encoding(data_cat, data_cat_m, self.all_levels, has_miss=False)
            self.n_classes = list(map(lambda x: len(x), self.all_levels))
            data_train = data_cat_enc
            data_train_m = data_cat_enc_miss
        ## normalize num
        if self.numindx:
            data_num = data_x[:, self.numindx]
            data_num_m = data_m[:, self.numindx]
            data_num_norm = data_num
            data_train = np.concatenate([data_train, data_num_norm], axis=1) if data_train.size else data_num_norm
            data_train_m = np.concatenate([data_train_m, data_num_m], axis=1) if data_train_m.size else data_num_m

        

        # Training Parameters
        learning_rate = self.lr
        num_steps1 = 200
        num_steps2 = 2
        batch_size = self.batch_size

        # Network Parameters
        num_input = data_train.shape[1]
        num_hidden_1 = data_train.shape[1] + self.theta  # 1st layer num features
        num_hidden_2 = data_train.shape[1] + 2 * self.theta # 2nd layer num features (the latent dim)
        num_hidden_3 = data_train.shape[1] + 3 * self.theta

        # A random value generator to initialize weights.
        random_normal = tf.initializers.RandomNormal()
        self.weights = {
            'encoder_h1': tf.Variable(random_normal([num_input, num_hidden_1])),
            'encoder_h2': tf.Variable(random_normal([num_hidden_1, num_hidden_2])),
            'encoder_h3': tf.Variable(random_normal([num_hidden_2, num_hidden_3])),
            'decoder_h1': tf.Variable(random_normal([num_hidden_3, num_hidden_2])),
            'decoder_h2': tf.Variable(random_normal([num_hidden_2, num_hidden_1])),
            'decoder_h3': tf.Variable(random_normal([num_hidden_1, num_input])),
        }
        self.biases = {
            'encoder_b1': tf.Variable(random_normal([num_hidden_1])),
            'encoder_b2': tf.Variable(random_normal([num_hidden_2])),
            'encoder_b3': tf.Variable(random_normal([num_hidden_3])),
            'decoder_b1': tf.Variable(random_normal([num_hidden_2])),
            'decoder_b2': tf.Variable(random_normal([num_hidden_1])),
            'decoder_b3': tf.Variable(random_normal([num_input])),
        }

        # Building the encoder
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate, decay=0.0)

        # multiple imputation
        # Start Training
        # Training phase 1
        loss_list = []
        pbar = tqdm(range(num_steps1))
        for i in pbar:
            # create mini batch
            indices = np.arange(no)
            np.random.shuffle(indices)
            for start_idx in range(0, no - batch_size + 1, batch_size):
                batch_idx = indices[start_idx:start_idx + batch_size]
                batch_x = data_train[batch_idx, :]
                batch_m = data_train_m[batch_idx, :]

                # Run optimization op (backprop) and cost op (to get loss value)
                l, y_hat = self.optimize_step(batch_x, batch_m)
                #pbar.set_description("loss at epoch {}: {:.3f}, phase 1".format(i, l))
                loss_list.append(l)

        imputed_data = self.decoder(self.encoder(data_train))
        imputed_data = data_train_m * data_train + (1 - data_train_m) * imputed_data

        # Training phase 2
        pbar = tqdm(range(num_steps2))
        for i in pbar:
            # create mini batch
            indices = np.arange(no)
            np.random.shuffle(indices)
            for start_idx in range(0, no - batch_size + 1, batch_size):
                batch_idx = indices[start_idx:start_idx + batch_size]
                batch_x = tf.gather(imputed_data, batch_idx, axis=0)
                batch_m = data_train_m[batch_idx, :]

                # Run optimization op (backprop) and cost op (to get loss value)
                l, y_hat = self.optimize_step(batch_x, batch_m)
                #pbar.set_description("loss at epoch {}, phase 2: {:.3f}".format(i, l))
                loss_list.append(l)

        return self

    def transform(self, X):
        """Imputes all missing values in X.

        Note that this is stochastic, and that if random_state is not fixed,
        repeated calls, or permuted input, will yield different results.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        Xt : array-like, shape (n_samples, n_features)
             The imputed input data.
        """

        #Code by George Paterakis
        #List of Lists -->> np.array with samples as rows and features as columns.

        data_x=np.transpose(np.array(X))
        no, dim = data_x.shape
        # initial imputation
        data_x = self._initial_imputation(data_x)

        data_m = 1-np.isnan(data_x)


        data_train = np.array([])
        data_train_m = np.array([])
        ## encode cat
        if self.catindx:
            data_cat = data_x[:, self.catindx]
            data_cat_m = data_m[:, self.catindx]
            data_cat_enc, data_cat_enc_miss = onehot_encoding(data_cat, data_cat_m, self.all_levels, has_miss=False)
            #n_classes = list(map(lambda x: len(x), self.all_levels))
            data_train = data_cat_enc
            data_train_m = data_cat_enc_miss
        ## normalize num
        if self.numindx:
            data_num = data_x[:, self.numindx]
            data_num_m = data_m[:, self.numindx]
            data_num_norm = data_num
            data_train = np.concatenate([data_train, data_num_norm], axis=1) if data_train.size else data_num_norm
            data_train_m = np.concatenate([data_train_m, data_num_m], axis=1) if data_train_m.size else data_num_m

        
        
        # get imputation
        imputed_data = self.decoder(self.encoder(data_train))
        imputed_data = data_train_m * data_train + (1 - data_train_m) * imputed_data

        # revert onehot and renormalize
        imputed = np.empty(shape=(no, dim))
        if self.catindx:
            imputed_cat = imputed_data[:, :data_cat_enc.shape[1]]
            imputed_cat = onehot_decoding(imputed_cat, data_cat_enc_miss, self.all_levels, has_miss=False)
            imputed[:,  self.catindx] = imputed_cat
        if  self.numindx:
            imputed_num = imputed_data[:, -data_num.shape[1]:]
            imputed[:, self.numindx] = imputed_num

        #Code by George Paterakis
        #Turn np.array for transform to List of Lists
        #Samples turn to columns , and columns to rows.

        X=np.transpose(np.array(imputed)).tolist()

        return X,self.new_names, self.new_vmaps

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
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]


        column_names = list(X_train.columns)

        if dataset == 'realdata\jad_audiology.csv' or dataset == 'realdata\MAR_10_molecular-biology_promoters.csv' or dataset == 'realdata\jad_primary-tumor.csv':
            tr= OrdinalEncoder(unknown_value=np.nan,handle_unknown="use_encoded_value")
            X_train = pd.DataFrame(tr.fit_transform(X_train))
            X_test = pd.DataFrame(tr.transform(X_test))
            Imputed_Train = X_train.values
            Imputed_Test = X_test.values
            categorical_features = column_names
            vmaps=dict(zip(categorical_features, ['' for i in categorical_features]))
        else:
            sclr = ColumnTransformer(
                transformers=[
                    ("std", 
                    StandardScaler(), 
                    [column_names.index(i) for i in column_names if i not in vmaps.keys()])],
                    remainder = 'passthrough'
            )

            Imputed_Train=sclr.fit_transform(X_train)
            Imputed_Test=sclr.transform(X_test)


        LL_train = np.transpose(Imputed_Train).tolist()
        LL_test  = np.transpose(Imputed_Test).tolist()

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

        
        #NP ARRAY TO DF
        
        X__train_imputed = pd.DataFrame(Imputed_Train,columns=column_names) 
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
    return error/5,total







for file_name in glob.glob('realdata/'+'*.csv'):
#for file_name in ['realdata/MAR_50_zoo.csv']:
#for file_name in ['realdata\\lymphoma_2classes.csv','realdata\\NewFuelCar.csv']: 
    print(file_name)
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
    elif file_name == 'realdata\jad_audiology.csv':
        categorical_features = ['age_gt_60', 'air', 'airBoneGap', 'ar_c', 'ar_u', 'bone', 'boneAbnormal', 'bser', 'history_buzzing', 'history_dizziness', 'history_fluctuating', 'history_fullness', 'history_heredity', 'history_nausea', 'history_noise', 'history_recruitment', 'history_ringing', 'history_roaring', 'history_vomiting', 'late_wave_poor', 'm_at_2k', 'm_cond_lt_1k', 'm_gt_1k', 'm_m_gt_2k', 'm_m_sn', 'm_m_sn_gt_1k', 'm_m_sn_gt_2k', 'm_m_sn_gt_500', 'm_p_sn_gt_2k', 'm_s_gt_500', 'm_s_sn', 'm_s_sn_gt_1k', 'm_s_sn_gt_2k', 'm_s_sn_gt_3k', 'm_s_sn_gt_4k', 'm_sn_2_3k', 'm_sn_gt_1k', 'm_sn_gt_2k', 'm_sn_gt_3k', 'm_sn_gt_4k', 'm_sn_gt_500', 'm_sn_gt_6k', 'm_sn_lt_1k', 'm_sn_lt_2k', 'm_sn_lt_3k', 'middle_wave_poor', 'mod_gt_4k', 'mod_mixed', 'mod_s_mixed', 'mod_s_sn_gt_500', 'mod_sn', 'mod_sn_gt_1k', 'mod_sn_gt_2k', 'mod_sn_gt_3k', 'mod_sn_gt_4k', 'mod_sn_gt_500', 'notch_4k', 'notch_at_4k', 'o_ar_c', 'o_ar_u', 's_sn_gt_1k', 's_sn_gt_2k', 's_sn_gt_4k', 'speech', 'static_normal', 'tymp', 'viith_nerve_signs', 'wave_V_delayed', 'waveform_ItoV_prolonged']
    elif file_name == 'realdata\MAR_50_churn.csv':
        categorical_features = ['international_plan','voice_mail_plan']
    else:
        categorical_features = []
    vmaps=dict(zip(categorical_features, ['' for i in categorical_features]))

    for ep in [500]:
        for theta in [7]:
            for drop in [0.5]:
                for lr in [0.001]:
                    error,total=loop(dataset=file_name,sep=';',na_values='?',outcome_Type='binaryClass',problem='C',vmaps=vmaps,parameter={"epochs":ep,"dropout":drop,"theta":theta,"lr":lr})
                    print('Error for ' + file_name+   ' : ' + str(error) + '  Time : ' + str(datetime.timedelta(seconds=total)) + ' Params  : ' + str({"epochs":ep,"dropout":drop,"theta":theta}) )


"""

def autoencoder_imputation(data_x, data_m, cat_index, num_index, all_levels, DAE_params, num_imputations):
    no, dim = data_x.shape
    # initial imputation
    data_x = initial_imputation(data_x, cat_index, num_index)

    data_train = np.array([])
    data_train_m = np.array([])
    ## encode cat
    if cat_index:
        data_cat = data_x[:, cat_index]
        data_cat_m = data_m[:, cat_index]
        data_cat_enc, data_cat_enc_miss = onehot_encoding(data_cat, data_cat_m, all_levels, has_miss=False)
        n_classes = list(map(lambda x: len(x), all_levels))
        data_train = data_cat_enc
        data_train_m = data_cat_enc_miss
    ## normalize num
    if num_index:
        data_num = data_x[:, num_index]
        data_num_m = data_m[:, num_index]
        data_num_norm, norm_parameters = normalization(data_num)
        data_train = np.concatenate([data_train, data_num_norm], axis=1) if data_train.size else data_num_norm
        data_train_m = np.concatenate([data_train_m, data_num_m], axis=1) if data_train_m.size else data_num_m

    # Training Parameters
    learning_rate = DAE_params["learning_rate"]
    num_steps1 = DAE_params["num_steps_phase1"]
    num_steps2 = DAE_params["num_steps_phase2"]
    batch_size = DAE_params["batch_size"]

    # Network Parameters
    num_input = data_train.shape[1]
    num_hidden_1 = data_train.shape[1] + DAE_params["theta"]  # 1st layer num features
    num_hidden_2 = data_train.shape[1] + 2 * DAE_params["theta"]  # 2nd layer num features (the latent dim)
    num_hidden_3 = data_train.shape[1] + 3 * DAE_params["theta"]

    # A random value generator to initialize weights.
    random_normal = tf.initializers.RandomNormal()

    weights = {
        'encoder_h1': tf.Variable(random_normal([num_input, num_hidden_1])),
        'encoder_h2': tf.Variable(random_normal([num_hidden_1, num_hidden_2])),
        'encoder_h3': tf.Variable(random_normal([num_hidden_2, num_hidden_3])),
        'decoder_h1': tf.Variable(random_normal([num_hidden_3, num_hidden_2])),
        'decoder_h2': tf.Variable(random_normal([num_hidden_2, num_hidden_1])),
        'decoder_h3': tf.Variable(random_normal([num_hidden_1, num_input])),
    }
    biases = {
        'encoder_b1': tf.Variable(random_normal([num_hidden_1])),
        'encoder_b2': tf.Variable(random_normal([num_hidden_2])),
        'encoder_b3': tf.Variable(random_normal([num_hidden_3])),
        'decoder_b1': tf.Variable(random_normal([num_hidden_2])),
        'decoder_b2': tf.Variable(random_normal([num_hidden_1])),
        'decoder_b3': tf.Variable(random_normal([num_input])),
    }

    # Building the encoder
    @tf.function
    def encoder(x):
        x_noise = tf.nn.dropout(x, 0.5)
        layer_1 = tf.nn.tanh(tf.add(tf.matmul(x_noise, weights['encoder_h1']),
                                       biases['encoder_b1']))
        layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                       biases['encoder_b2']))
        layer_3 = tf.nn.tanh(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                       biases['encoder_b3']))
        return layer_3

    # Building the decoder
    @tf.function
    def decoder(x):
        layer_1 = tf.nn.tanh(tf. add(tf.matmul(x, weights['decoder_h1']),
                                       biases['decoder_b1']))
        layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                       biases['decoder_b2']))
        layer_3 = tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                         biases['decoder_b3'])
        col_index = 0
        empty_G_out = True
        # apply softmax to each categorical variable
        if cat_index:
            empty_G_out = False
            output = tf.nn.softmax(layer_3[:, :n_classes[0]])
            col_index = n_classes[0]
            for j in range(1, len(n_classes)):
                output = tf.concat(values=[output, tf.nn.softmax(layer_3[:, col_index:col_index + n_classes[j]])], axis=1)
                col_index += n_classes[j]
        # apply sigmoid to all numerical variables
        if num_index:
            out_num = tf.nn.sigmoid(layer_3[:, col_index:])
            output = tf.concat(values=[output, out_num], axis=1) if not empty_G_out else out_num
        return output

    # sum up loss for each categorical variable
    @tf.function
    def dae_loss(y_pred, y_true, mask):
        loss = 0
        current_ind = 0
        # categorical loss
        if cat_index:
            for j in range(len(n_classes)):
                mask_current = mask[:, current_ind:current_ind + n_classes[j]]
                y_pred_current = y_pred[:, current_ind:current_ind + n_classes[j]]
                y_true_current = y_true[:, current_ind:current_ind + n_classes[j]]
                loss += -tf.reduce_mean(
                    input_tensor=mask_current * y_true_current * tf.math.log(mask_current * y_pred_current + 1e-8)) / tf.reduce_mean(
                    input_tensor=mask_current)
                current_ind += n_classes[j]
        # numerical loss
        if num_index:
            mask_current = mask[:, current_ind:]
            y_pred_current = y_pred[:, current_ind:]
            y_true_current = y_true[:, current_ind:]
            loss += tf.reduce_mean((mask_current * y_true_current - mask_current * y_pred_current)**2) / tf.reduce_mean(mask_current)
        return loss

    # optimizer
    @tf.function
    def optimize_step(batch_x, batch_m):
        with tf.GradientTape() as g:
            y_hat = decoder(encoder(batch_x))
            l = dae_loss(y_hat, batch_x, batch_m)

        trainable_variables = list(weights.values()) + list(biases.values())

        gradients = g.gradient(l, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))
        return l, y_hat

    optimizer = tf.optimizers.Adam(lr=learning_rate, decay=0.0)

    # multiple imputation
    imputed_list = []
    for l in range(num_imputations):
        # Start Training
        # Training phase 1
        loss_list = []
        pbar = tqdm(range(num_steps1))
        for i in pbar:
            # create mini batch
            indices = np.arange(no)
            np.random.shuffle(indices)
            for start_idx in range(0, no - batch_size + 1, batch_size):
                batch_idx = indices[start_idx:start_idx + batch_size]
                batch_x = data_train[batch_idx, :]
                batch_m = data_train_m[batch_idx, :]

                # Run optimization op (backprop) and cost op (to get loss value)
                l, y_hat = optimize_step(batch_x, batch_m)
                pbar.set_description("loss at epoch {}: {:.3f}, phase 1".format(i, l))
                loss_list.append(l)

        imputed_data = decoder(encoder(data_train))
        imputed_data = data_train_m * data_train + (1 - data_train_m) * imputed_data

        # Training phase 2
        pbar = tqdm(range(num_steps2))
        for i in pbar:
            # create mini batch
            indices = np.arange(no)
            np.random.shuffle(indices)
            for start_idx in range(0, no - batch_size + 1, batch_size):
                batch_idx = indices[start_idx:start_idx + batch_size]
                batch_x = tf.gather(imputed_data, batch_idx, axis=0)
                batch_m = data_train_m[batch_idx, :]

                # Run optimization op (backprop) and cost op (to get loss value)
                l, y_hat = optimize_step(batch_x, batch_m)
                pbar.set_description("loss at epoch {}, phase 2: {:.3f}".format(i, l))
                loss_list.append(l)

        # get imputation
        imputed_data = decoder(encoder(imputed_data))
        imputed_data = data_train_m * data_train + (1 - data_train_m) * imputed_data

        # revert onehot and renormalize
        imputed = np.empty(shape=(no, dim))
        if cat_index:
            imputed_cat = imputed_data[:, :data_cat_enc.shape[1]]
            imputed_cat = onehot_decoding(imputed_cat, data_cat_enc_miss, all_levels, has_miss=False)
            imputed[:, cat_index] = imputed_cat
        if num_index:
            imputed_num = imputed_data[:, -data_num.shape[1]:]
            imputed_num = renormalization(imputed_num.numpy(), norm_parameters)
            imputed[:, num_index] = imputed_num
        imputed_list.append(imputed)
    return imputed_list, loss_list
"""