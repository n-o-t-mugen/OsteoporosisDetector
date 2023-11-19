import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def preprocess_image(image_path, target_size):
    img = Image.open(image_path)
    img = img.convert('L')  
    img = img.resize(target_size, Image.ANTIALIAS)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255.0  
    return img_array

def predict(model, image_path):
    img_array = preprocess_image(image_path, model.input_shape[1:3])
    prediction = model.predict(img_array)
    return prediction[0][0]

def main():
    model_path = '/Users/mruthunjai_govindaraju/Desktop/untitled folder/model.h5'
    test_images_directory = '/Users/mruthunjai_govindaraju/Downloads/archive (2)/osteoporosis/osteoporosis'

    model = load_model(model_path)

    for image_name in os.listdir(test_images_directory):
        image_path = os.path.join(test_images_directory, image_name)
        prediction = predict(model, image_path)
        print(f"{image_name}: {'Osteoporosis' if prediction > 0.5 else 'Normal'}")

        
        img = Image.open(image_path)
        plt.imshow(img, cmap='gray')
        plt.title(f"Prediction: {'Osteoporosis' if prediction > 0.5 else 'Normal'}")
        plt.show()

if __name__ == "__main__":
    main()
