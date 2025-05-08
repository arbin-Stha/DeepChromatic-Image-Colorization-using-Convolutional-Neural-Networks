import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model

def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0  # Normalize to [0, 1]
    x_test = x_test.astype('float32') / 255.0
    return x_train, y_train, x_test, y_test

def plot_sample_images(images, labels):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(f'Label: {labels[i][0]}')
        plt.axis('off')
    plt.show()

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()

    # Plot some sample images from the CIFAR-10 dataset
    plot_sample_images(x_train, y_train)

    # Create and compile the model
    model = create_model()
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)

    # Evaluate the model on the test set
    results = model.evaluate(x_test, y_test, verbose=2)
    print(f'\nTest evaluation results: {results}')  # Check what is being returned

    test_loss, test_acc = results
    print(f'\nTest Loss: {test_loss}, Test Accuracy: {test_acc}')

    # Save the model
    model.save('cifar10_cnn_model.h5')
