import onnxruntime as ort

# Load the ONNX model
sess_options = ort.SessionOptions()
sess_options.optimized_model_filepath = "models/rf_optimized_onnx_model.onnx"

# Enable all optimizations
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Create a session with the optimizations enabled
session = ort.InferenceSession("models/rf_onnx_model.onnx", sess_options)

