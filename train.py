import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

def train_random_forest_classifier(final_df):

    X = final_df.drop(columns=['Index'], axis=1).values
    y = final_df['Index'].values

    X_train, X_test, y_train,  y_test = train_test_split(X,y, test_size=0.1, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    rf.fit(X_train,y_train)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_pred, y_test)

    print(f'Accuracy: {accuracy*100}%') 

    with open('models/random_forest_classifier_model.pkl', 'wb') as f:
        pickle.dump(rf, f)


if __name__ == '__main__':

    print('Run in pipeline!')