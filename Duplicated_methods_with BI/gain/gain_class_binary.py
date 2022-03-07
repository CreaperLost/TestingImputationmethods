# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# Necessary packages
import tensorflow as tf
##IF USING TF 2 use following import to still use TF < 2.0 Functionalities
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
tf.compat.v1.disable_v2_behavior()
# Packages
import numpy as np
from tqdm.notebook import tqdm_notebook as tqdm


import seaborn as sns
sns.set()
import numpy as np

import numpy as np
from tqdm import tqdm

from utils import normalization, renormalization, rounding
from utils import xavier_init
from utils import binary_sampler, uniform_sampler, sample_batch_index
from sklearn.impute import MissingIndicator


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


        self.names = names
        self.new_names = names
        self.vmaps = vmaps
        self.new_vmaps = vmaps


        #The indexes for categorical features in feature list.
        self.catindx = [names.index(i) for i in vmaps.keys()]
        self.numindx = [names.index(i) for i in names if i not in vmaps.keys()]
        self.cat_names = [i for i in vmaps.keys()]
        self.num_names = [i for i in names if i not in vmaps.keys()]
  
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
        
        ## GAIN architecture   
        # Input placeholders
        # Data vector
        self.X = tf.compat.v1.placeholder(tf.float32, shape = [None, dim])
        # Mask vector 
        self.M = tf.compat.v1.placeholder(tf.float32, shape = [None, dim])
        # Hint vector
        self.H = tf.compat.v1.placeholder(tf.float32, shape = [None, dim])
        
        # Discriminator variables
        D_W1 = tf.Variable(xavier_init([dim*2, h_dim])) # Data + Hint as inputs
        D_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
        
        D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
        D_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
        
        D_W3 = tf.Variable(xavier_init([h_dim, dim]))
        D_b3 = tf.Variable(tf.zeros(shape = [dim]))  # Multi-variate outputs
        
        theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
        
        #Generator variables
        # Data + Mask as inputs (Random noise is in missing components)
        G_W1 = tf.Variable(xavier_init([dim*2, h_dim]))  
        G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
        
        G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
        G_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
        
        G_W3 = tf.Variable(xavier_init([h_dim, dim]))
        G_b3 = tf.Variable(tf.zeros(shape = [dim]))
        
        theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]


        ## GAIN functions
        # Generator
        def generator(x,m):
            # Concatenate Mask and Data
            inputs = tf.concat(values = [x, m], axis = 1) 
            G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
            G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)   
            # MinMax normalized output
            G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3) 
            return G_prob
            
        # Discriminator
        def discriminator(x, h):
            # Concatenate Data and Hint
            inputs = tf.concat(values = [x, h], axis = 1) 
            D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)  
            D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
            D_logit = tf.matmul(D_h2, D_W3) + D_b3
            D_prob = tf.nn.sigmoid(D_logit)
            return D_prob

        ## GAIN structure
        # Generator
        self.G_sample = generator(self.X, self.M)
        
        # Combine with observed data
        Hat_X = self.X * self.M + self.G_sample * (1-self.M)
        
        # Discriminator
        D_prob = discriminator(Hat_X, self.H)
        
        ## GAIN loss
        D_loss_temp = -tf.reduce_mean(self.M * tf.compat.v1.log(D_prob + 1e-8) \
                                        + (1-self.M) * tf.compat.v1.log(1. - D_prob + 1e-8)) 
        
        G_loss_temp = -tf.reduce_mean((1-self.M) * tf.compat.v1.log(D_prob + 1e-8))
        
        MSE_loss = \
        tf.reduce_mean((self.M * self.X - self.M * self.G_sample)**2) / tf.reduce_mean(self.M)
        
        D_loss = D_loss_temp
        G_loss = G_loss_temp + self.alpha * MSE_loss 
        
        ## GAIN solver
        D_solver = tf.compat.v1.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
        G_solver = tf.compat.v1.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
        
        ## Iterations
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        
        # Start Iterations 
        #tqdm(range(self.iterations))
        for it in tqdm(range(self.iterations)):    
        
            # Sample batch
            batch_idx = sample_batch_index(no, self.batch_size)
            X_mb = norm_data_x[batch_idx, :]  
            M_mb = data_m[batch_idx, :]  
            # Sample random vectors  
            Z_mb = uniform_sampler(0, 0.01, self.batch_size, dim) 
            # Sample hint vectors
            H_mb_temp = binary_sampler(self.hint_rate, self.batch_size, dim)
            H_mb = M_mb * H_mb_temp
            
            # Combine random vectors with observed vectors
            X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
            
            _, D_loss_curr = self.sess.run([D_solver, D_loss_temp], 
                                    feed_dict = {self.M: M_mb, self.X: X_mb,self.H: H_mb})
            _, G_loss_curr, MSE_loss_curr = \
            self.sess.run([G_solver, G_loss_temp, MSE_loss],
                    feed_dict = {self.X: X_mb, self.M: M_mb, self.H: H_mb})
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
        X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
            
        imputed_data = self.sess.run([self.G_sample], feed_dict = {self.X: X_mb, self.M: M_mb})[0]
        
        imputed_data = data_m * norm_data_x + (1-data_m) * imputed_data
        
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


  

  

