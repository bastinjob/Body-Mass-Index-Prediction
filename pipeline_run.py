from preprocess import preprocess_data
from train import train_random_forest_classifier
from test import validate_test_data

print('Reading and Preprocessing dataset')

df = preprocess_data(dataset_path='data/bmi_train.csv')

print('Commencing training of random forest classifier')

train_random_forest_classifier(df)

validate_test_data('data/bmi_validation.csv','models/random_forest_classifier_model.pkl','preprocess_models/scaler.pkl')

print('Run Successful!')