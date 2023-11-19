import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class OsteoporosisModel:
    """
    CNN model for classifying bone X-rays for osteoporosis.
    """
    def __init__(self, input_shape=(256, 256, 1)):
        self.input_shape = input_shape
        self.model = self._build_model()

    def _build_model(self):
        inputs = layers.Input(shape=self.input_shape)

        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1, activation='sigmoid')(x)

        model = models.Model(inputs=inputs, outputs=x)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        return model

    def train(self, data_directory, batch_size=32, epochs=10):
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

        train_generator = datagen.flow_from_directory(
            data_directory,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='binary',
            subset='training',
            color_mode='grayscale'
        )

        validation_generator = datagen.flow_from_directory(
            data_directory,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='binary',
            subset='validation',
            color_mode='grayscale'
        )

        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            epochs=epochs
        )
        return history

    def evaluate(self, data_directory, batch_size=32):
        datagen = ImageDataGenerator(rescale=1./255)

        test_generator = datagen.flow_from_directory(
            data_directory,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='binary',
            color_mode='grayscale',
            shuffle=False
        )

        loss, accuracy = self.model.evaluate(test_generator, steps=test_generator.samples // batch_size)
        print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    def plot_history(self, history):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def save_model(self, filepath):
        self.model.save(filepath)

def main():
    data_directory = "/Users/mruthunjai_govindaraju/Downloads/archive (2)"  
    model = OsteoporosisModel()
    history = model.train(data_directory, batch_size=32, epochs=10)
    model.evaluate(data_directory)
    model.plot_history(history)
    model.save_model('/Users/mruthunjai_govindaraju/Desktop/untitled folder/model.h5')

if __name__ == "__main__":
    main()