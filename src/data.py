from keras.datasets import mnist
from keras.utils import to_categorical

def load_data():
    (train_data_x, train_labels_y), (test_data_x, test_labels_y) = mnist.load_data()

    # Normalización
    x_train = train_data_x.reshape(60000, 28*28).astype('float32') / 255
    x_test = test_data_x.reshape(10000, 28*28).astype('float32') / 255

    # Codificación one-hot
    y_train = to_categorical(train_labels_y, 10)
    y_test = to_categorical(test_labels_y, 10)

    return x_train, y_train, x_test, y_test
