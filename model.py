import numpy as np
import tensorflow as tf
import matplotlib as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models


class OsteoporosisModel:
    """
    A Convolutional Neural Network (CNN) model for classifying bone X-rays for osteoporosis.
    The model architecture, training, evaluation, and plotting of results are included in this class.
    """

    def __init__(self, input_shape=(256, 256, 1)):
        """
        Initialize the model with a given input shape.

        Parameters:
        input_shape (tuple): The shape of the input images. Default is (256, 256, 1) for grayscale images.
        """
        self.input_shape = input_shape
        self.model = self._build_model()

    def _build_model(self):
        """
        Private method to build the CNN model.

        Returns:
        model: A compiled TensorFlow Keras model.
        """
        # Model architecture
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
        """
        Train the model with data from the specified directory.

        Parameters:
        data_directory (str): Path to the directory containing training data.
        batch_size (int): Batch size for training. Default is 32.
        epochs (int): Number of epochs to train the model. Default is 10.

        Returns:
        history: Training history object containing training and validation metrics.
        """
        # Data preparation and augmentation
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

        # Training data generator
        train_generator = datagen.flow_from_directory(
            data_directory,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='binary',
            subset='training',
            color_mode='grayscale'
        )

        # Validation data generator
        validation_generator = datagen.flow_from_directory(
            data_directory,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='binary',
            subset='validation',
            color_mode='grayscale'
        )

        # Model training
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            epochs=epochs
        )

        return history

    def evaluate(self, data_directory, batch_size=32):
        """
        Evaluate the model performance on test data.

        Parameters:
        data_directory (str): Path to the directory containing test data.
        batch_size (int): Batch size for evaluation. Default is 32.
        """
        # Data preparation
        datagen = ImageDataGenerator(rescale=1./255)
        test_generator = datagen.flow_from_directory(
            data_directory,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='binary',
            color_mode='grayscale',
            shuffle=False
        )

        # Model evaluation
        loss, accuracy = self.model.evaluate(test_generator, steps=test_generator.samples // batch_size)
        print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    def plot_history(self, history):
        """
        Plot training history including accuracy and loss over epochs.

        Parameters:
        history: Training history object to plot.
        """
        # Plotting accuracy and loss
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
        """
        Save the trained model to a file.

        Parameters:
        filepath (str): Path where the model should be saved.
        """
        self.model.save(filepath)

def main():
    """
    Main function to execute the model training, evaluation, and plotting.
    """
    data_directory = "path_to_data_directory"  
    model = OsteoporosisModel()
    history = model.train(data_directory, batch_size=32, epochs=10)
    model.evaluate(data_directory)
    model.plot_history(history)
    model.save_model('model.h5')

if __name__ == "__main__":
    main()
