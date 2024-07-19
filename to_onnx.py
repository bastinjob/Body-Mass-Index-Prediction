import skl2onnx 
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
import pickle


with open('models/random_forest_classifier_model.pkl', 'rb') as file:
    skl_model = pickle.load(file)

# Determine the number of features
try:
    n_features = skl_model.n_features_in_
except AttributeError:
    # If the model doesn't have n_features_in_ attribute, set it manually
    n_features = 3 

initial_type = [('float_input',FloatTensorType([None,n_features]))]
onnx_model = convert_sklearn(skl_model, initial_types=initial_type)

with open('models/rf_onnx_model.onnx','wb') as file:
    file.write(onnx_model.SerializeToString())