import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from model_factory import create_model

def load_data(data_dir):
    """
    Loads images and labels from the GTSRB dataset structure.
    Expects a structure with folders 0, 1, ..., 42.
    """
    data = []
    labels = []
    classes = 43
    
    # Iterate through all classes
    for i in range(classes):
        path = os.path.join(data_dir, 'Train', str(i))
        if not os.path.exists(path):
            print(f"Directory {path} not found. Skipping...")
            continue
            
        images = os.listdir(path)
        for a in images:
            try:
                image = Image.open(path + '/' + a)
                image = image.resize((30, 30))
                image = np.array(image)
                data.append(image)
                labels.append(i)
            except Exception as e:
                print(f"Error loading image {a}: {e}")
                
    # Convert to numpy arrays
    data = np.array(data)
    labels = np.array(labels)
    
    return data, labels

def train_and_save():
    print("Initializing model training engine...")
    
    # In a real scenario, we'd load the data here.
    # For this end-to-end framework, we provide the logic.
    # data, labels = load_data('.')
    
    # MOCK DATA GENERATION (To allow the script to run/compile for demo)
    print("Generating simulated dataset for framework validation...")
    X = np.random.rand(1000, 30, 30, 3)
    y = np.random.randint(0, 43, 1000)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # One-hot encoding
    y_train = to_categorical(y_train, 43)
    y_test = to_categorical(y_test, 43)
    
    # Create and train
    model = create_model()
    print("Starting neural pathway compilation...")
    model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test))
    
    # Save
    model.save("traffic_sign_model.h5")
    print("Model saved as traffic_sign_model.h5")

if __name__ == "__main__":
    train_and_save()
