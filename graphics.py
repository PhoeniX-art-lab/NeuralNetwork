import matplotlib.pyplot as plt


def graphics(history):
    plt.plot(history.history['accuracy'],
             label='Доля верных ответов на обучающем наборе')
    plt.plot(history.history['val_accuracy'],
             label='Доля правильных ответов на проверочном наборе')
    plt.xlabel('Эпоха обучения')
    plt.ylabel('Доля верных ответов')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'],
             label='Ошбика на обучающем наборе')
    plt.plot(history.history['val_loss'],
             label='Ошбика на проверочном наборе')
    plt.xlabel('Эпоха обучение')
    plt.ylabel('Ошибка')
    plt.legend()
    plt.show()
