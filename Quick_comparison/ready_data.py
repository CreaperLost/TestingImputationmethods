import glob
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import numpy  as np




for file_name in glob.glob('realdata/'+'*.csv'):
    data=pd.read_csv(file_name,sep=';',na_values='?')
    print(file_name,data.info())
    if file_name == 'realdata\colleges_aaup.csv':
        categorical_features = ["State", "Type", "binaryClass"]
    elif file_name == 'realdata\colleges_usnews.csv':
        categorical_features = ["State", "binaryClass"]
    elif file_name == 'realdata\heart-h.csv':
        categorical_features = ["sex","chest_pain","fbs","restecg","exang","slope","thal", "binaryClass"]
    elif file_name == 'realdata\kdd_coil_1.csv':
        categorical_features = ["season","river_size","fluid_velocity"]
    elif file_name == 'realdata\kdd_el_nino-small.csv':
        categorical_features = ['binaryClass']
    elif file_name == 'realdata\meta.csv':
        categorical_features = ['DS_Name','Alg_Name']
    elif file_name == 'realdata\schizo.csv':
        categorical_features = ['target','sex','binaryClass']
    elif file_name == 'realdata\water-treatment.csv':
        categorical_features = ['binaryClass']
    elif file_name == 'realdata\Moneyball.csv':
        categorical_features = ['Team','League']
    else:
        continue


    categorical_transformer = OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=np.nan)
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
        ],remainder='drop'
    )
    data[categorical_features]  = preprocessor.fit_transform(data)
    print(data.head())
    data.to_csv(file_name,sep=';',na_rep='?',index=False)
