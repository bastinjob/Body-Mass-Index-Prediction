import numpy as np
import pandas as pd
import pickle


def validate_test_data(test_data_path,model_path, scaler_path):

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    test_df = pd.read_csv(test_data_path)

    #test_df_encoded = pd.get_dummies(test_df, columns=['Gender'])

    test_df['Gender_encoded'] = [1. if gender == 'Male' else 0. for gender in test_df['Gender'].values]

    scaled_hw = pd.DataFrame(scaler.transform(test_df[['Height', 'Weight']]), columns = ['Height', 'Weight'])

    final_test_df = pd.concat([scaled_hw, test_df.drop(columns=['Height','Weight','Gender'],axis=1)],axis=1)

    preds = model.predict(final_test_df.values)

    dict_bmi = {0: 'Extremly Weak',
                1:'Weak',
                2 :'Normal',
                3:'Overweight',
                4:'Obesity',
                5:'Extremly Obese'
                }
    
    obesity_level = [dict_bmi[pred] for pred in preds]

    final_test_df['BMI'] = preds
    final_test_df['Obesity_Level'] = obesity_level
    final_test_df.to_csv('Inferences/Estimated_BMI.csv')


    print('Inference Succesful!')



if __name__ == '__main__':

    validate_test_data('data/bmi_validation.csv','models/random_forest_classifier_model.pkl','preprocess_models/scaler.pkl')