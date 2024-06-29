from preprocess import preprocess_data
from train import train_random_forest_classifier

print('Reading and Preprocessing dataset')

df = preprocess_data(dataset_path='data/bmi_train.csv')

print('Commencing training of random forest classifier')

train_random_forest_classifier(df)

print('Run Successful!')