from keras.models import Sequential
from keras.layers import Dense, Input

def build_model():
    model = Sequential([
        Input(shape=(28*28,)),
        Dense(512, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compilar el modelo
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
