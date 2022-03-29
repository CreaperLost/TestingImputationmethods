# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from gain_v2_utils import xavier_init, binary_sampler, uniform_sampler, onehot_encoding, onehot_decoding, normalization, renormalization

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np
import glob
import datetime
import numpy as np
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,StandardScaler
import numpy as np
import glob
import numpy as np
import time
from sklearn.compose import ColumnTransformer


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

        self.theta_G= None
        self.names = names
        self.new_names = names
        self.vmaps = vmaps
        self.new_vmaps = vmaps


        #The indexes for categorical features in feature list.
        self.catindx = [names.index(i) for i in vmaps.keys()]
        self.numindx = [names.index(i) for i in names if i not in vmaps.keys()]
        self.cat_names = [i for i in vmaps.keys()]
        self.num_names = [i for i in names if i not in vmaps.keys()]
        self.n_classes = []
        self.all_levels = []

    @tf.function
    def generator(self,x,m):
            # Concatenate Mask and Data
            inputs = tf.concat(values = [x, m], axis = 1)
            
            G_h1 = tf.nn.leaky_relu(tf.matmul(inputs, self.G[0]) + self.G[3])
            G_h2 = tf.nn.leaky_relu(tf.matmul(G_h1, self.G[1]) + self.G[4])
            G_logit = tf.matmul(G_h2, self.G[2]) + self.G[5]

            col_index = 0
            empty_G_out = True
            # apply softmax to each categorical variable
            if self.catindx:
                empty_G_out = False
                G_out = tf.nn.softmax(G_logit[:, :self.n_classes[0]])
                col_index = self.n_classes[0]
                for j in range(1, len(self.n_classes)):
                    G_out = tf.concat(values=[G_out, tf.nn.softmax(G_logit[:, col_index:col_index + self.n_classes[j]])], axis=1)
                    col_index += self.n_classes[j]
            # apply sigmoid to all numerical variables
            if self.numindx:
                G_out_num = tf.nn.sigmoid(G_logit[:, col_index:])
                G_out = tf.concat(values=[G_out, G_out_num], axis=1) if not empty_G_out else G_out_num
            return G_out
    # Discriminator
    @tf.function
    def discriminator(self,x, h):
            # Concatenate Data and Hint
            inputs = tf.concat(values = [x, h], axis = 1)
            D_h1 = tf.nn.leaky_relu(tf.matmul(inputs, self.theta_D[0]) + self.theta_D[3])
            D_h2 = tf.nn.leaky_relu(tf.matmul(D_h1, self.theta_D[1]) + self.theta_D[4])
            D_logit = tf.matmul(D_h2, self.theta_D[2]) + self.theta_D[5]
            D_prob = tf.nn.sigmoid(D_logit)
            return D_prob

    # loss function
    @tf.function
    def gain_Dloss(self,D_prob, mask):
            D_loss_temp = -tf.reduce_mean(mask * tf.math.log(D_prob + 1e-7) +
                                        (1 - mask) * tf.math.log(1. - D_prob + 1e-7))
            D_loss = D_loss_temp
            return D_loss

    @tf.function
    def gain_Gloss(self,sample, G_sample, D_prob, mask, n_classes):
            G_loss_temp = -tf.reduce_mean((1 - mask) * tf.math.log(D_prob + 1e-7))
            reconstruct_loss = 0

            # categorical loss
            current_ind = 0
            if self.catindx:
                for j in range(len(n_classes)):
                    M_current = mask[:, current_ind:current_ind + n_classes[j]]
                    G_sample_temp = G_sample[:, current_ind:current_ind + n_classes[j]]
                    X_temp = sample[:, current_ind:current_ind + n_classes[j]]
                    reconstruct_loss += -tf.reduce_mean(M_current * X_temp * tf.math.log(M_current * G_sample_temp + 1e-7)) / tf.reduce_mean(
                        M_current)
                    current_ind += n_classes[j]
            # numerical loss
            if self.numindx:
                M_current = mask[:, current_ind:]
                G_sample_temp = G_sample[:, current_ind:]
                X_temp = sample[:, current_ind:]
                reconstruct_loss += tf.reduce_mean((M_current * X_temp - M_current * G_sample_temp) ** 2) / tf.reduce_mean(
                    M_current)
            return G_loss_temp, reconstruct_loss

    def fit(self,X):
        #Code by George Paterakis
        #List of Lists -->> np.array with samples as rows and features as columns.
        data_x=np.transpose(np.array(X))
        # Define mask matrix
        data_m = 1-np.isnan(data_x)
        self.all_levels = [np.unique(x)[~np.isnan(np.unique(x))] for x in data_x[:, self.catindx].T]
        data_train = np.array([])
        data_train_m = np.array([])
        # preprocess categorical variables
        if self.catindx:
            data_cat = data_x[:, self.catindx]
            data_cat_m = data_m[:, self.catindx]
            data_cat_enc, data_cat_enc_miss = onehot_encoding(data_cat, data_cat_m, self.all_levels, has_miss=True)
            data_cat_enc = np.nan_to_num(data_cat_enc, 0)
            data_train = data_cat_enc
            data_train_m = data_cat_enc_miss
            self.n_classes = list(map(lambda x: len(x), self.all_levels))
        # preprocess numerical variables
        if self.numindx:
            data_num = data_x[:, self.numindx]
            data_num_m = data_m[:, self.numindx]
            #data_num_norm, norm_parameters = normalization(data_num)
            data_num_norm = np.nan_to_num(data_num, 0)
            data_train = np.concatenate([data_train, data_num_norm], axis=1) if data_train.size else data_num_norm
            data_train_m = np.concatenate([data_train_m, data_num_m], axis=1) if data_train_m.size else data_num_m

        
        # Other parameters
        no, dim = data_x.shape
        input_dim = data_train.shape[1]

        # Hidden state dimensions
        h_Gdim = int(input_dim)
        h_Ddim = int(input_dim)

        ## GAIN architecture
        # Discriminator variables
        D_W1 = tf.Variable(xavier_init([input_dim*2, h_Ddim])) # Data + Hint as inputs
        D_b1 = tf.Variable(tf.zeros(shape = [h_Ddim]))

        D_W2 = tf.Variable(xavier_init([h_Ddim, h_Ddim]))
        D_b2 = tf.Variable(tf.zeros(shape = [h_Ddim]))

        D_W3 = tf.Variable(xavier_init([h_Ddim, input_dim]))
        D_b3 = tf.Variable(tf.zeros(shape = [input_dim]))  # Multi-variate outputs

        self.theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

        #Generator variables
        # Data + Mask as inputs (Random noise is in missing components)
        G_W1 = tf.Variable(xavier_init([input_dim*2, h_Gdim]))
        G_b1 = tf.Variable(tf.zeros(shape = [h_Gdim]))

        G_W2 = tf.Variable(xavier_init([h_Gdim, h_Gdim]))
        G_b2 = tf.Variable(tf.zeros(shape = [h_Gdim]))

        G_W3 = tf.Variable(xavier_init([h_Gdim, input_dim]))
        G_b3 = tf.Variable(tf.zeros(shape = [input_dim]))

        self.theta_G = [G_W1, G_W3, G_b1, G_b3]
        self.G = [G_W1,G_W2, G_W3, G_b1,G_b2, G_b3]
        

        # optimizer
        @tf.function
        def optimize_step(self,X_mb, M_mb, H_mb, n_classes):
            with tf.GradientTape() as g:
                # Generator
                G_sample = self.generator(X_mb, M_mb)
                # Combine with observed data
                Hat_X = X_mb * M_mb + G_sample * (1 - M_mb)
                # Discriminator
                D_prob = self.discriminator(Hat_X, H_mb)
                D_loss = self.gain_Dloss(D_prob, M_mb)

            Dgradients = g.gradient(D_loss, self.theta_D)
            D_solver.apply_gradients(zip(Dgradients, self.theta_D))

            for i in range(3):
                with tf.GradientTape() as g:
                    # Generator
                    G_sample = self.generator(X_mb, M_mb)
                    # Combine with observed data
                    Hat_X = X_mb * M_mb + G_sample * (1 - M_mb)
                    # Discriminator
                    D_prob = self.discriminator(Hat_X, H_mb)
                    G_loss_temp, reconstructloss = self.gain_Gloss(X_mb, G_sample, D_prob, M_mb, n_classes)
                    G_loss = G_loss_temp + self.alpha*reconstructloss
                Ggradients = g.gradient(G_loss, self.theta_G)
                G_solver.apply_gradients(zip(Ggradients, self.theta_G))
            return D_loss, G_loss_temp, reconstructloss

        ## GAIN solver
        D_solver = tf.optimizers.Adam()
        G_solver = tf.optimizers.Adam()


        # Start Iterations
        Gloss_list = []
        Dloss_list = []
        pbar = tqdm(range(self.iterations))
        for i in pbar:
            # create mini batch
            indices = np.arange(no)
            np.random.shuffle(indices)
            for start_idx in range(0, no - self.batch_size + 1, self.batch_size):
                batch_idx = indices[start_idx:start_idx + self.batch_size]
                X_mb = data_train[batch_idx, :]
                M_mb = data_train_m[batch_idx, :]

                # Sample random vectors
                Z_mb = uniform_sampler(0, 0.01, self.batch_size, input_dim)
                # Sample hint vectors
                H_mb_temp = binary_sampler(self.hint_rate, self.batch_size, input_dim)
                H_mb = M_mb * H_mb_temp

                # Combine random vectors with observed vectors
                X_mb = M_mb * X_mb + (1-M_mb) * Z_mb

                X_mb=tf.convert_to_tensor(X_mb,dtype=tf.float32)
                M_mb=tf.convert_to_tensor(M_mb,dtype=tf.float32)
                H_mb=tf.convert_to_tensor(H_mb,dtype=tf.float32)
                
                D_loss_curr, G_loss_curr, reconstructloss = optimize_step(self,X_mb, M_mb, H_mb, self.n_classes)
                Gloss_list.append(G_loss_curr)
                Dloss_list.append(D_loss_curr)
                pbar.set_description("D_loss: {:.3f}, G_loss: {:.3f}, Reconstruction loss: {:.3f}".format(D_loss_curr.numpy(),
                                                                                                        G_loss_curr.numpy(),
                                                                                                        reconstructloss.numpy()))
        return self
               
    def transform(self,X):
        #Code by George Paterakis
        #List of Lists -->> np.array with samples as rows and features as columns.
        data_x=np.transpose(np.array(X))

        # Define mask matrix
        data_m = 1-np.isnan(data_x)
        no_final, dim_final = data_x.shape
        
        data_train = np.array([])
        data_train_m = np.array([])
        # preprocess categorical variables
        if self.catindx:
            data_cat = data_x[:, self.catindx]
            data_cat_m = data_m[:, self.catindx]
            data_cat_enc, data_cat_enc_miss = onehot_encoding(data_cat, data_cat_m, self.all_levels, has_miss=True)
            data_cat_enc = np.nan_to_num(data_cat_enc, 0)
            data_train = data_cat_enc
            data_train_m = data_cat_enc_miss
            #self.n_classes = list(map(lambda x: len(x), self.all_levels))
        # preprocess numerical variables
        if self.numindx:
            data_num = data_x[:, self.numindx]
            data_num_m = data_m[:, self.numindx]
            #data_num_norm, norm_parameters = normalization(data_num)
            data_num_norm = np.nan_to_num(data_num, 0)
            data_train = np.concatenate([data_train, data_num_norm], axis=1) if data_train.size else data_num_norm
            data_train_m = np.concatenate([data_train_m, data_num_m], axis=1) if data_train_m.size else data_num_m


        X_mb = data_train
        M_mb = data_train_m
        # Other parameters
        no, dim = X_mb.shape
    
        # Hidden state dimensions
        Z_mb = uniform_sampler(0, 0.01, no, dim)
        
        X_mb = M_mb * X_mb + (1-M_mb) * Z_mb

        X_mb=tf.convert_to_tensor(X_mb,dtype=tf.float32)
        M_mb=tf.convert_to_tensor(M_mb,dtype=tf.float32)
        Z_mb=tf.convert_to_tensor(Z_mb,dtype=tf.float32)

        imputed_data = self.generator(X_mb, M_mb)
        imputed_data = M_mb * data_train + (1-M_mb) * imputed_data

        # revert onehot and renormalize
        imputed = np.empty(shape=(no_final, dim_final))
        if self.catindx:
            imputed_cat = imputed_data[:, : data_cat_enc.shape[1]]
            imputed_cat = onehot_decoding(imputed_cat, data_cat_enc_miss, self.all_levels, has_miss=False)
            imputed[:, self.catindx] = imputed_cat
        if self.numindx:
            imputed_num = imputed_data[:, - data_num.shape[1]:]
            imputed[:, self.numindx] = imputed_num

        imputed_data = np.transpose(np.array(imputed)).tolist()

        return imputed_data,self.names,self.vmaps




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
        return
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


        if dataset == 'realdata\jad_audiology.csv' or dataset == 'realdata\MAR_10_molecular-biology_promoters.csv' or dataset == 'realdata\jad_primary-tumor.csv':
            tr= OrdinalEncoder(unknown_value=np.nan,handle_unknown="use_encoded_value")
            X_train = pd.DataFrame(tr.fit_transform(X_train))
            X_test = pd.DataFrame(tr.transform(X_test))

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
        imputer = Gain(parameters=parameter,vmaps=vmaps,names=column_names)


        Methods_Impute = imputer.fit(LL_train)

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
    return error/5,total


for file_name in glob.glob('realdata/'+'*.csv'):
#for file_name in ['realdata\MAR_50_zoo.csv']:
#for file_name in ['realdata\MCAR_50_Boston.csv','realdata\MCAR_50_Australian.csv']:

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
    print(categorical_features)

    for ep in [1000]:
        for hi_p in [0.9]:
            for alp in [1]:
                error , total = loop(dataset=file_name,sep=';',na_values='?',outcome_Type='binaryClass',problem='C',vmaps=vmaps,parameter={"iterations":ep,"hint_rate":hi_p,"alpha":alp,"Binary_Indicator":True})
                print('Error for ' + file_name+   ' : ' + str(error) + '  Time : ' + str(datetime.timedelta(seconds=total)) + ' Params  : ' + str({"epochs":ep,"hint_rate":hi_p,"alpha":alp}) )
                with open('gain_res+bi.txt', 'a') as the_file:
                    the_file.write('Error for ' + file_name+   ' : ' + str(error) + '  Time : ' + str(datetime.timedelta(seconds=total)) + ' Params  : ' + str({"epochs":ep,"hint_rate":hi_p,"alpha":alp}) + '\n' )



  
