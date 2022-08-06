import numpy as np
from sklearn.utils import  check_random_state
from sklearn.utils.validation import  check_is_fitted
from sklearn.utils._mask import _get_mask
from sklearn.impute._base import SimpleImputer
import pandas as pd
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
import rpy2.robjects.numpy2ri as numpy2ri
from rpy2.robjects import r, pandas2ri
#from encoder import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

class SoftImpute:
    
    def __init__(self,parameters: dict, names: list, vmaps: dict,
                 missing_values=np.nan):

        package_names =  {'softImpute'}

        if not all(rpackages.isinstalled(x) for x in package_names):
            utils = rpackages.importr('utils')
            utils.chooseCRANmirror(ind=1)

            packnames_to_install = [x for x in package_names if not rpackages.isinstalled(x)]
            
            if len(packnames_to_install) >0:
                utils.install_packages(robjects.StrVector(packnames_to_install))
        
        self.softImpute = rpackages.importr('softImpute')
        self.base = rpackages.importr('base')


        pandas2ri.activate()

        self.enc = OneHotEncoder(handle_unknown='ignore',sparse=False)

        self.estimator = None
        self.nPcs = parameters.get("nPcs",5)
        self.lamda = parameters.get("lambda",30)
        self.names = names
        self.new_names = names
        self.vmaps = vmaps
        self.new_vmaps = vmaps

        numpy2ri.activate()

        #The indexes for categorical features in feature list.
        self.catindx = [names.index(i) for i in vmaps.keys()]
        self.numindx = [names.index(i) for i in names if i not in vmaps.keys()]
        self.cat_names = [i for i in vmaps.keys()]
        self.num_names = [i for i in names if i not in vmaps.keys()]
        self.missing_values = missing_values


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

        #if numericals
        if len(self.numindx) >0 :
            X_num  = X[:,self.numindx]
            X_conc = X_num
        
        

        #if categoricals
        if len(self.catindx) >0 :
            #Do one hot encoding to cat variables.
            X_cat = X[:,self.catindx]
            X_cat = pd.DataFrame(X_cat,columns=self.vmaps.keys())
            self.enc.fit(X_cat)
            X_cat=self.enc.transform(X_cat)
            X_conc = X_cat


        #If mixed type then concat
        if len(self.catindx) >0 and len(self.numindx) >0:
            X_conc  = np.concatenate((X_num ,X_cat),axis=1)

        X_conc = robjects.r('''
                        function(X){ 
                        X[X=="NaN"] <- NA 
                        X }''')(X_conc)

        self.estimator = self.softImpute.softImpute(X_conc,self.nPcs,self.lamda,type="svd")

        X_r = self.softImpute.complete(X_conc,self.estimator,unscale=True)
        
        #if mixed slice.
        if len(self.catindx) >0 and len(self.numindx) >0:
            X_num_sliced,X_cat_sliced = X_r[:,:len(self.numindx)] , X_r[:,len(self.numindx):]
            X_cat_sliced=self.enc.inverse_transform(X_cat_sliced)
            X_r  = np.concatenate((X_num_sliced ,X_cat_sliced),axis=1)
            self.new_names = self.num_names + self.cat_names
        elif len(self.catindx) >0:
            X_r=self.enc.inverse_transform(X_r)


        X=np.transpose(np.array(X_r)).tolist()


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
        #check_is_fitted(self)

        #Code by George Paterakis
        #List of Lists -->> np.array with samples as rows and features as columns.
        X=np.transpose(np.array(X))

        #if numericals
        if len(self.numindx) >0 :
            X_num  = X[:,self.numindx]
            X_conc = X_num

        #if categoricals
        if len(self.catindx) >0 :
            #Do one hot encoding to cat variables.
            X_cat = X[:,self.catindx]
            X_cat = pd.DataFrame(X_cat,columns=self.vmaps.keys())
            X_cat=self.enc.transform(X_cat)
            X_conc = X_cat
        
        #If mixed type then concat
        if len(self.catindx) >0 and len(self.numindx) >0:
            X_conc  = np.concatenate((X_num ,X_cat),axis=1)


        X_conc = robjects.r('''
                        function(X){ 
                        X[X=="NaN"] <- NA 
                        X }''')(X_conc)


        X_r= self.softImpute.complete(X_conc,self.estimator,unscale=True)

        #if mixed slice.
        if len(self.catindx) >0 and len(self.numindx) >0:
            X_num_sliced,X_cat_sliced = X_r[:,:len(self.numindx)] , X_r[:,len(self.numindx):]
            X_cat_sliced=pd.DataFrame(X_cat_sliced).round()
            X_cat_sliced=self.enc.inverse_transform(X_cat_sliced)
            X_r  = np.concatenate((X_num_sliced ,X_cat_sliced),axis=1)
            self.new_names = self.num_names + self.cat_names
        elif len(self.catindx) >0:
            X_r=self.enc.inverse_transform(X_r)

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