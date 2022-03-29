from interpretableai import iai
import pandas as pd
import numpy as np
import pandas as pd






class Optimal_knn_Imputation():

    def __init__(self,parameters: dict, names: list, vmaps: dict) -> None:
        # System parameters
        self.knn_k = parameters.get('knn_k',5)

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
        
        self.method = iai.OptKNNImputationLearner(knn_k=self.knn_k,treat_unknown_level_missing=True,show_progress=False,random_seed=1).fit(na_data)

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


