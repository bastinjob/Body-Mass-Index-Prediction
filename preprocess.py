'''
loads, preprocesses the data and return it
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

def preprocess_data(dataset_path='data/bmi_train.csv'):

    df = pd.read_csv(dataset_path)

    # One-hot encode the Gender column
    df_encoded = pd.get_dummies(df, columns=['Gender'])

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on the height and weight columns and transform them
    scaled_features = scaler.fit_transform(df_encoded[['Height', 'Weight']])

    # Create a DataFrame with the scaled features
    df_scaled = pd.DataFrame(scaled_features, columns=['Height_scaled', 'Weight_scaled'])

    # Concatenate with the original DataFrame (if needed)
    df_final = pd.concat([df_encoded.drop(columns=['Height','Weight'],axis=1), df_scaled], axis=1)

    #print(df_final)

    return df_final


if __name__ == '__main__':

    preprocess_data()

