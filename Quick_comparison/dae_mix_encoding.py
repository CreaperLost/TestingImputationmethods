import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

def onehot_encoding(data, data_m, all_levels, has_miss=False):
    no, dim = data.shape

    data_filled = np.nan_to_num(data.copy(), 0)
    data_enc = np.empty(shape=(no, np.sum([len(x) for x in all_levels])), dtype=np.float32)
    data_m_enc = np.empty(shape=(no, np.sum([len(x) for x in all_levels])), dtype=np.float32)
    col_idx = 0
    for j in range(dim):
        colj_nlevel = len(all_levels[j])
        colj = data_filled[:, j].astype(np.int)
        miss_j = np.repeat(data_m[:, j].reshape([-1, 1]), colj_nlevel,axis=1)
        enc_j = np.eye(colj_nlevel)[colj]
        if has_miss:
            enc_j[miss_j == 0] = np.nan
        data_enc[:, col_idx:(col_idx+colj_nlevel)] = enc_j
        data_m_enc[:, col_idx:(col_idx + colj_nlevel)] = miss_j
        col_idx += colj_nlevel
    return data_enc, data_m_enc

def onehot_decoding(data_enc, data_m_enc, all_levels, has_miss=False):
    col_idx = 0
    no = data_enc.shape[0]
    dim = len(all_levels)

    miss_enc = data_m_enc
    data = np.empty(shape=(no, dim), dtype=np.float32)
    for j in range(dim):
        colj_level = len(all_levels[j])
        data_enc_j = data_enc[:, col_idx:(col_idx + colj_level)]
        data_j = np.argmax(data_enc_j, axis=1).astype(np.float32)
        data_m_j = miss_enc[:, col_idx]
        if has_miss:
            data_j[data_m_j == 0] = np.nan
        data[:, j] = data_j
        col_idx += colj_level
    return data

class Autoencoder(nn.Module):
    def __init__(self, dim , theta,dropout):
        super(Autoencoder, self).__init__()
        self.dim = dim
        
        self.drop_out = nn.Dropout(p=dropout)
        
        self.encoder = nn.Sequential(
            nn.Linear(dim+(theta*0), dim+(theta*1)),
            nn.Tanh(),
            nn.Linear(dim+(theta*1), dim+(theta*2)),
            nn.Tanh(),
            nn.Linear(dim+(theta*2), dim+(theta*3))
        )
            
        self.decoder = nn.Sequential(
            nn.Linear(dim+(theta*3), dim+(theta*2)),
            nn.Tanh(),
            nn.Linear(dim+(theta*2), dim+(theta*1)),
            nn.Tanh(),
            nn.Linear(dim+(theta*1), dim+(theta*0))
        )
        
    def forward(self, x):
        x = x.view(-1, self.dim)
        x_missed = self.drop_out(x)
        
        z = self.encoder(x_missed)
        out = self.decoder(z)
        
        out = out.view(-1, self.dim)
        
        return out



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




    def forward(self, x):
        x = x.view(-1, self.dim)
        x_missed = self.drop_out(x)
        
        z = self.encoder(x_missed)
        out = self.decoder(z)
        
        out = out.view(-1, self.dim)
        
        return out


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
        device = torch.device('cpu')
        
        self.initial_imputer_Mean = None
        self.initial_imputer_Mode = None


        #Code by George Paterakis
        #List of Lists -->> np.array with samples as rows and features as columns.
        X=np.transpose(np.array(X))



        data_m = 1-np.isnan(X)

        X_r = self._initial_imputation(X)

        self.all_levels = [np.unique(x)[~np.isnan(np.unique(x))] for x in X_r[:, self.catindx].T]

        
        #if numericals
        if len(self.numindx) >0 :
            X_num  = X_r[:,self.numindx]
            X_conc = X_num
        
        
        #if categoricals
        if len(self.catindx) >0 :
            #Do one hot encoding to cat variables.
            X_cat = X_r[:,self.catindx]
            #X_cat = pd.DataFrame(X_cat,columns=self.vmaps.keys())
            
            data_cat_enc, data_cat_enc_miss = onehot_encoding(X_cat, X_cat, self.all_levels, has_miss=False)
            self.n_classes = list(map(lambda x: len(x), self.all_levels))
            #self.onehot.fit(X_cat)
            #X_cat=self.onehot.transform(X_cat)
            X_conc = data_cat_enc
            
            

        #If mixed type then concat
        if len(self.catindx) >0 and len(self.numindx) >0:
            X_conc  = np.concatenate((X_num ,X_cat),axis=1)
        

        train_data = torch.from_numpy(X_conc).float()

        train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=min(self.batch_size, X.shape[0]),shuffle=True)


        cost_list = []
        early_stop = False

        self.dim = X_conc.shape[1]

        self.model = Autoencoder(dim = self.dim,theta = self.theta,dropout=self.drop_out).to(device)
        self.loss = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), momentum=0.99, lr=self.lr, nesterov=True)


        for epoch in range(self.epochs):
            
            total_batch = len(train_data)//min(self.batch_size, X.shape[0])
            
            for i, batch_data in enumerate(train_loader):
                
                batch_data = batch_data.to(device)
                
                reconst_data = self.model(batch_data)
                cost = self.loss(reconst_data, batch_data)
                
                self.optimizer.zero_grad()
                cost.backward()
                self.optimizer.step()
                        
                #if (i+1) % (total_batch//2) == 0:
                #   print('Epoch [%d/%d], lter [%d/%d], Loss: %.6f'%(epoch+1, self.epochs, i+1, total_batch, cost.item()))
                    
                # early stopping rule 1 : MSE < 1e-06
                if cost.item() < 1e-06 :
                    early_stop = True
                    break
                    
                cost_list.append(cost.item())

            if early_stop :
                break

        #Evaluate
        self.model.eval()
        filled_data = self.model(train_data.to(device))
        filled_data_train = filled_data.cpu().detach().numpy()


        #if mixed slice.
        if len(self.catindx) >0 and len(self.numindx) >0:
            X_num_sliced,X_cat_sliced = filled_data_train[:,:len(self.numindx)] , filled_data_train[:,len(self.numindx):]
            #X_cat_sliced=self.onehot.inverse_transform(X_cat_sliced)
            X_cat_sliced = onehot_decoding(filled_data_train, data_cat_enc_miss, self.all_levels, has_miss=False)
            filled_data_train  = np.concatenate((X_num_sliced ,X_cat_sliced),axis=1)
            self.new_names = self.num_names + self.cat_names
        elif len(self.catindx) >0:
            filled_data_train = onehot_decoding(filled_data_train, data_cat_enc_miss, self.all_levels, has_miss=False)
            #filled_data_train=self.onehot.inverse_transform(filled_data_train)
            

        X=np.transpose(np.array(filled_data_train)).tolist()


        return X

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
        device = torch.device('cpu')
        #check_is_fitted(self)

        #Code by George Paterakis
        #List of Lists -->> np.array with samples as rows and features as columns.
        X=np.transpose(np.array(X))

        data_m = 1-np.isnan(X)

        X_filled = self._initial_imputation(X)


        #if numericals
        if len(self.numindx) >0 :
            X_num  = X_filled[:,self.numindx]
            X_conc = X_num

        #if categoricals
        if len(self.catindx) >0 :
            #Do one hot encoding to cat variables.
            X_cat = X_filled[:,self.catindx]
            X_cat, data_cat_enc_miss = onehot_encoding(X_cat, X_cat, self.all_levels, has_miss=False)
            #X_cat = pd.DataFrame(X_cat,columns=self.vmaps.keys())
            #X_cat=self.onehot.transform(X_cat)
            X_conc = X_cat
            


        #If mixed type then concat
        if len(self.catindx) >0 and len(self.numindx) >0:
            X_conc  = np.concatenate((X_num ,X_cat),axis=1)

        X_filled = torch.from_numpy(X_conc).float()
        #Evaluate
        self.model.eval()

        #Transform Test set
        filled_data = self.model(X_filled.to(device))
        filled_data_test = filled_data.cpu().detach().numpy()
        X_r = filled_data_test

        
        #if mixed slice.
        if len(self.catindx) >0 and len(self.numindx) >0:
            X_num_sliced,X_cat_sliced = filled_data_test[:,:len(self.numindx)] , filled_data_test[:,len(self.numindx):]
            #X_cat_sliced=pd.DataFrame(X_cat_sliced) 
            X_cat_sliced = onehot_decoding(X_cat_sliced, data_cat_enc_miss, self.all_levels, has_miss=False)
            #X_cat_sliced=self.onehot.inverse_transform(X_cat_sliced)
            X_r  = np.concatenate((X_num_sliced ,X_cat_sliced),axis=1)
            self.new_names = self.num_names + self.cat_names
        elif len(self.catindx) >0:
            X_r = onehot_decoding(filled_data_test, data_cat_enc_miss, self.all_levels, has_miss=False)
            #X_r=self.onehot.inverse_transform(filled_data_test)
         
        #Code by George Paterakis
        #Turn np.array for transform to List of Lists
        #Samples turn to columns , and columns to rows.

        X=np.transpose(np.array(X_r)).tolist()

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