import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from PIL import Image
import cv2

# Mapping of labels to traffic sign names (GTSRB classes)
CLASSES = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing veh over 3.5 tons',
    11: 'Right-of-way at intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Veh > 3.5 tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve left',
    20: 'Dangerous curve right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End speed + passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End no passing veh > 3.5 tons'
}

def create_model(input_shape=(30, 30, 3)):
    """
    Creates the CNN model architecture for traffic sign recognition.
    """
    model = Sequential([
        # First layer
        Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=input_shape),
        Conv2D(filters=32, kernel_size=(5, 5), activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(rate=0.25),

        # Second layer
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(rate=0.25),

        # Flatten and Dense layers
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(rate=0.5),
        Dense(43, activation='softmax')  # 43 classes
    ])

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def preprocess_image(image):
    """
    Prepares an image for prediction.
    - Resize to (30, 30)
    - Normalize
    - Reshape to add batch dimension (1, 30, 30, 3)
    """
    try:
        # If image is path, open it. If it's PIL, use it.
        if isinstance(image, str):
            image = Image.open(image)
        
        # Ensure RGB
        image = image.convert('RGB')
        
        # Resize to match model input
        image = image.resize((30, 30))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Reshape to (1, 30, 30, 3)
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

if __name__ == "__main__":
    # Test model creation
    model = create_model()
    model.summary()
