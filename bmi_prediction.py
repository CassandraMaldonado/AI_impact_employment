# BMI Predictor

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from mtcnn import MTCNN
import base64
from PIL import Image
from io import BytesIO

# Importing our custom VGG implementation
from vgg_face import get_vgg_face_model, preprocess_input

# Image size required from VGG as the input
IMAGE_SIZE = (224, 224)

class BMIPredictor:   
    def __init__(self, model_path=None):
        self.detector = MTCNN()
        self.model = None
        
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model = load_model(model_path)

    # Creating a BMI prediction model based on the VGG    
    def create_model(self):
        # Get the VGG-Face model
        base_model = get_vgg_face_model(include_top=False, input_shape=(224, 224, 3))
        
        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        # Create a new model with unique layer names
        inputs = tf.keras.Input(shape=(224, 224, 3))
        vgg_output = base_model(inputs)
        
        # Add custom regression layers with explicit names
        x = tf.keras.layers.Flatten(name='bmi_flatten')(vgg_output)
        x = tf.keras.layers.Dense(512, activation='relu', name='bmi_dense1')(x)
        x = tf.keras.layers.Dropout(0.5, name='bmi_dropout1')(x)
        x = tf.keras.layers.Dense(128, activation='relu', name='bmi_dense2')(x)
        x = tf.keras.layers.Dropout(0.5, name='bmi_dropout2')(x)
        predictions = tf.keras.layers.Dense(1, activation='linear', name='bmi_output')(x)
        
        # Create the model with the new architecture
        model = tf.keras.Model(inputs=inputs, outputs=predictions)
        
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        
        self.model = model
        return model

# Detecting the faces in the image and preprocessing it 
    def preprocess_face_image(self, image_path=None, img=None, method='mtcnn'):
        try:
            # Loading the image
            if image_path:
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Error loading image: {image_path}")
                    return None
                
                # BGR to RGB for VGG model requirements
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Checking the image
            if img is None:
                print("No image provided")
                return None
            
            # Converting it to RGB 
            if len(img.shape) == 2:  
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:  
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            if method == 'mtcnn':
                # MTCNN for face detection
                faces = self.detector.detect_faces(img)
                if not faces:
                    print(f"No face detected in the image")
                    # Using the entire image if theres no face detected
                    face_img = cv2.resize(img, IMAGE_SIZE)
                else:
                    # Getting the largest face
                    face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
                    x, y, w, h = face['box']
                    
                    x, y = max(0, x), max(0, y)
                    w = min(w, img.shape[1] - x)
                    h = min(h, img.shape[0] - y)
                    
                    # Cropping the face
                    face_img = img[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, IMAGE_SIZE)
            else:
                # Use the entire image
                face_img = cv2.resize(img, IMAGE_SIZE)
            
            # Preprocess
            face_img = preprocess_input(face_img, version=1)
            
            return face_img
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None
    
# Converting the text encoded image data back into an actual picture
    def preprocess_base64_image(self, base64_string):
        try:
            img_data = base64.b64decode(base64_string)
            img = Image.open(BytesIO(img_data))
            img = np.array(img)
            
            # Processing the face
            face_img = self.preprocess_face_image(img=img, method='mtcnn')
            
            if face_img is None:
                return None
                
            return np.expand_dims(face_img, axis=0)
            
        except Exception as e:
            print(f"Error processing base64 image: {str(e)}")
            return None
    
    def load_data(self, data_path, image_folder):
        print("Loading and preprocessing dataset...")
        
        # Data
        df = pd.read_csv(data_path)
        
        X = []
        y = []
        
        total = len(df)
        for idx, row in df.iterrows():
            print(f"Processing image {idx+1}/{total}: {row['name']}")
            
            image_path = os.path.join(image_folder, row['name'])
            
            # Checking if the image exists
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found")
                continue
            
            # Preprocessing the image
            img = self.preprocess_face_image(image_path=image_path)
            if img is not None:
                X.append(img)
                y.append(row['bmi'])
        
        return np.array(X), np.array(y)

# Training the model with the dataset
    def train(self, X, y, epochs=50, batch_size=4, validation_split=0.2, callbacks=None):
        if self.model is None:
            self.create_model()
            
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1,
            callbacks=callbacks
        )
        
        return history

# Predicting the BMI value and category from the image.
    def predict_bmi(self, image_path=None, img=None, base64_string=None):
        if self.model is None:
            raise ValueError("Model not loaded. Call create_model() or load a model first.")
        
        # Processing the image
        if base64_string:
            processed_image = self.preprocess_base64_image(base64_string)
        elif image_path or img is not None:
            face_img = self.preprocess_face_image(image_path=image_path, img=img)
            if face_img is None:
                return None, None
            processed_image = np.expand_dims(face_img, axis=0)
        else:
            raise ValueError("No image provided. Provide image_path, img or base64_string.")
        
        if processed_image is None:
            return None, None
            
        # Making the prediction
        predicted_bmi = float(self.model.predict(processed_image)[0][0])
        
        # BMI categories
        category = 'Unknown'
        if predicted_bmi < 18.5:
            category = 'Underweight'
        elif 18.5 <= predicted_bmi < 25:
            category = 'Normal weight'
        elif 25 <= predicted_bmi < 30:
            category = 'Overweight'
        else:
            category = 'Obesity'
        
        return predicted_bmi, category
    
    def save_model(self, model_path):
        if self.model is None:
            raise ValueError("No model to save. Call create_model() first.")
            
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

# Saving the training history plot
    def save_training_plot(self, history, filename='training_history.png'):

        plt.figure(figsize=(12, 4))
        
        # Training and validation loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        # Training and validation mean absolute error
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.title('Model Mean Absolute Error')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

if __name__ == "__main__":
    # Example usage
    predictor = BMIPredictor()
    
    # Create the model
    model = predictor.create_model()
    
    # Load data
    data_path = '/Users/casey/Desktop/UChicago/ML2/Code Final/data.csv'
    image_folder = '/Users/casey/Documents/GitHub/Face_to_BMI/Images'
    X, y = predictor.load_data(data_path, image_folder)
    
    # Train the model
    history = predictor.train(X, y, epochs=10, batch_size=4)
    
    # Save the model
    predictor.save_model('bmi_model.h5')
    
    # Predict BMI from an image
    bmi, category = predictor.predict_bmi(image_path='data/test_image.jpg')
    print(f"Predicted BMI: {bmi}, Category: {category}")
    # Save training history plot
    predictor.save_training_plot(history, 'training_history.png')