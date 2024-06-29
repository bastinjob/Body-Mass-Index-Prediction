from preprocess import preprocess_data
from train import train_random_forest_classifier

df = preprocess_data(dataset_path='data/bmi_train.csv')
train_random_forest_classifier(df)

print('Run Successful!')