import glob
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import numpy  as np


data=pd.read_csv('realdata\kdd_coil_1.csv',sep=';',na_values='?',decimal=",")

print(data.head(),data.info())
#data.to_csv('realdata\kdd_coil_1.csv',sep=';',na_rep='?',index=False)