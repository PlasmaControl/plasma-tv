import numpy as np
from pathlib import Path
from sklearn.externals import joblib

# Constants
MODEL_DIR = Path("../../models/")
PYTORCH = 'pytorch'
SKLEARN = 'sklearn'

# Load Models
def list_models(model_dir = MODEL_DIR):
    return [str(model_path) for model_path in model_dir.glob("*")]
    
def load_model(model_path, model_type = SKLEARN):
    if model_type == SKLEARN:
        return joblib.load(model_path)
    else: 
        raise ValueError("Unsupported model type")

# Preprocessing Functions
def preprocess(input_data):
    return input_data

# Prediction Functions
def predict(model, input_data, model_type = SKLEARN):
    input_data = preprocess(input_data)
    if model_type == SKLEARN:
        return model.predict(input_data)
    else:
        raise ValueError("Unsupported model type")

# Example Functions
if __name__ == "__main__":
    # Load models
    models = list_models()
    model = load_model(models[0], SKLEARN)

    # Example data (replace with actual data)
    example_input = np.random.rand(1, 10)

    # Make predictions
    prediction = predict(model, example_input, SKLEARN)
    print(prediction)