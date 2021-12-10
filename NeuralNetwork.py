from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing import image_dataset_from_directory
from consts import *


class NeuralNetwork:
    def __init__(self):
        self.path = f'D:/python/Dataset/mnist-{mnist_number}'
        self.train_dataset = image_dataset_from_directory(f'D:/python/Dataset/mnist-{mnist_number}',
                                                          color_mode='grayscale',
                                                          subset='training',
                                                          seed=42,
                                                          validation_split=0.1,
                                                          # 10% будут использоваться в качестве проверочного набора
                                                          batch_size=batch_size,
                                                          image_size=image_size)
        # создание проверочного набора
        self.validation_dataset = image_dataset_from_directory(f'D:/python/Dataset/mnist-{mnist_number}',
                                                               subset='validation',
                                                               color_mode='grayscale',
                                                               seed=42,
                                                               validation_split=0.1,
                                                               batch_size=batch_size,
                                                               image_size=image_size
                                                               )
        self.test_dataset = image_dataset_from_directory('D:/python/Dataset/mnist_png/testing',
                                                         color_mode='grayscale',
                                                         batch_size=batch_size,
                                                         image_size=image_size)

        self.model = Sequential([
            Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2), strides=2),
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            Flatten(),  # преобразует тензор в вектор
            Dense(128, activation='sigmoid'),
            Dense(10, activation='softmax')
        ])
