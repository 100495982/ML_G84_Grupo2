
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("Datasets/attrition_availabledata_09.csv")

# Drop the constant columns
df.drop(columns=["EmployeeCount", "Over18", "StandardHours"])

# Encode categorical and ordinal columsn



target = "Attrition"

"""
Imputation for missing values:
Numerical: SimpleImputer(strategy='median')
Categorical: SimpleImputer(strategy='most_frequent')
"""


def convert_to_numeric(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            ...

"""
Ordinal attributes to numerical:
BusinessTravel

"""