import numpy as np
import pandas as pd
# Read the dataset from csv file
# dataset is as type as dataframe which can't be used directly
dataset = pd.read_csv('dataset/Data.csv')

# Extract lines from the dataframe
X = dataset.iloc[:, : -1].values
Y = dataset.iloc[:, 3].values

# Preprocessing
# Replace NaN value to the mean of the colomn
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
# Use 3 colomns of X to train the Imputer
imputer = imputer.fit(X[:, 1:3])
# Replace the values of X with the trained Imputer
X[:, 1:3] = imputer.transform(X[:, 1:3])

# change labels to numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# https://tree.rocks/python/sklearn-explain-onehotencoder-use/
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Split dataset to training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)

# Normalization
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
