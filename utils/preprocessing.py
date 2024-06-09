import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def data_loader(data):
    df=pd.DataFrame(data.data, columns=data.feature_names)
    df['label']=data.target
    return df
def separate(df, label,test_size,random_state):
    X=df.drop(label, axis=1)
    y=df[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
def preprocess(train, test):
    scalar=StandardScaler()
    train_std=scalar.fit_transform(train)
    test_std=scalar.transform(test)
    return train_std, test_std
