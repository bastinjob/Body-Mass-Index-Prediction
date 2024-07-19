import numpy as np
import time
import pickle
import onnx
import onnxruntime as ort

n_features=3
#load the model
with open('models/random_forest_classifier_model.pkl', 'rb') as file:
    skl_model = pickle.load(file)
    
# Generate 100 input samples
X_test = np.random.rand(100, n_features).astype(np.float32)

# Measure inference time for the original scikit-learn model
start_time = time.time()
original_predictions = skl_model.predict(X_test)
original_inference_time = time.time() - start_time


# Load the optimized model
ort_session = ort.InferenceSession("models/rf_optimized_onnx_model.onnx")

# Measure inference time for the optimized ONNX model
ort_inputs = {ort_session.get_inputs()[0].name: X_test}
start_time = time.time()
ort_outs = ort_session.run(None, ort_inputs)
optimized_inference_time = time.time() - start_time

# Print the inference times
print("Original model inference time for 100 samples: {:.4f} seconds".format(original_inference_time))
print("Optimized ONNX model inference time for 100 samples: {:.4f} seconds".format(optimized_inference_time))

# Optionally, verify that the predictions match
# Note: Depending on the model and optimization, slight differences might occur due to floating-point precision.
optimized_predictions = ort_outs[0]
print("Original model predictions (first 5 samples):", original_predictions[:5])
print("Optimized model predictions (first 5 samples):", optimized_predictions[:5])