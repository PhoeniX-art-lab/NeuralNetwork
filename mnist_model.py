import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from NeuralNetwork import *
from graphics import *

obj = NeuralNetwork()
while mnist_number <= 5:
    if mnist_number != 1:
        obj.__init__()
    print(f'\n----Using path: {obj.path}\n')
    obj.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    history = obj.model.fit(obj.train_dataset,
                            validation_data=obj.validation_dataset,
                            epochs=17,
                            verbose=1)
    scores = obj.model.evaluate(obj.test_dataset, verbose=1)
    print('Доля верных ответов на тестовых данных, в процентах: ', round(scores[1] * 100, 4))
    accuracy.append(round(scores[1] * 100, 4))
    obj.model.save('mnist_model.h5')
    graphics(history)

    mnist_number += 1

print(f'Точность нейросети для 1-ого датасета: {accuracy[0]}\n'
      f'Точность нейросети для 2-ого датасета: {accuracy[1]}\n'
      f'Точность нейросети для 3-ого датасета: {accuracy[2]}\n'
      f'Точность нейросети для 4-ого датасета: {accuracy[3]}\n'
      f'Точность нейросети для 5-ого датасета: {accuracy[4]}')
print(f'Лучшая выборка для обучения нейросети #{np.argmax(accuracy) + 1}.\n'
      f'Точность данной выборки: {accuracy[np.argmax(accuracy)]}')

# Теперь докажем, почему СНС лучше нейросети обратного распространения
print('Возбмем наилучший датасет из полученных и на его основе обучим нейросеть обратного распространения')
mnist_number = np.argmax(accuracy)

model = Sequential([
    Flatten(input_shape=(28, 28, 1)),  # преобразует тензор в вектор
    Dense(256, activation='sigmoid'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(obj.train_dataset,
                    validation_data=obj.validation_dataset,
                    epochs=17,
                    verbose=1)

scores = model.evaluate(obj.test_dataset, verbose=1)
print('Доля верных ответов на тестовых данных, в процентах: ', round(scores[1] * 100, 4))
accuracy.append(round(scores[1] * 100, 4))

graphics(history)

print(f'Точность нейросети обратного распространения: {accuracy[0]}%\n')
