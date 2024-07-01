'''
loads, preprocesses the data and return it
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import OneHotEncoder
import pickle

def preprocess_data(dataset_path='data/bmi_train.csv'):

    df = pd.read_csv(dataset_path)

    # One-hot encode the Gender column
    #df_encoded = pd.get_dummies(df, columns=['Gender'])

    df['Gender_encoded'] = [1. if gender == 'Male' else 0. for gender in df['Gender'].values]


    # Concatenate with the original DataFrame (if needed)
    df_final = df.drop(columns=['Gender'],axis=1)

    print(df_final)


    return df_final


if __name__ == '__main__':

    preprocess_data()

