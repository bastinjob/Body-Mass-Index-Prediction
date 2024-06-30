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

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on the height and weight columns and transform them
    scaled_features = scaler.fit_transform(df[['Height', 'Weight']])

    # Create a DataFrame with the scaled features
    df_scaled = pd.DataFrame(scaled_features, columns=['Height_scaled', 'Weight_scaled'])

    # Concatenate with the original DataFrame (if needed)
    df_final = pd.concat([df.drop(columns=['Height','Weight', 'Gender'],axis=1), df_scaled], axis=1)

    print(df_final)

    with open('preprocess_models/scaler.pkl','wb') as f:
        pickle.dump(scaler, f)


    return df_final


if __name__ == '__main__':

    preprocess_data()

