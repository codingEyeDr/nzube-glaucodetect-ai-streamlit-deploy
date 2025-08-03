import os
import tensorflow as tf
from tensorflow.keras import layers

# 1. VERIFY FILE PATH
MODEL_PATH = "NzubeGlaucoma_AI_Predictor.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {os.path.abspath(MODEL_PATH)}")

# 2. ENHANCED CUSTOM LAYER
class CustomInputLayer(layers.InputLayer):
    def __init__(self, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['input_shape'] = kwargs['batch_shape'][1:]  # Convert batch_shape to input_shape
            kwargs.pop('batch_shape')
        elif 'input_shape' not in kwargs:
            kwargs['input_shape'] = (224, 224, 3)  # Default shape matching your error
        
        kwargs.pop('sparse', None)
        kwargs.pop('ragged', None)
        super().__init__(**kwargs)

# 3. LOAD WITH DEBUGGING
try:
    print(f"Loading model from: {os.path.abspath(MODEL_PATH)}")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            'InputLayer': CustomInputLayer,
            'DTypePolicy': tf.keras.mixed_precision.Policy,
            'GlorotUniform': tf.keras.initializers.GlorotUniform,
            'Zeros': tf.keras.initializers.Zeros
        },
        compile=False
    )
    print("Model loaded successfully! Layer summary:")
    model.summary()
    
    # 4. SAVE NEW VERSION
    output_path = "NzubeGlaucoma_AI_Predictor_FINAL.h5"
    model.save(output_path, save_format="h5")
    print(f"Converted model saved to: {os.path.abspath(output_path)}")
    
except Exception as e:
    print(f"CONVERSION FAILED: {str(e)}")
    print("\nTROUBLESHOOTING:")
    print("1. Verify TensorFlow version == 2.15.0")
    print("2. Check model file integrity with HDF Viewer")
    print("3. Try opening in Colab with GPU runtime")