def train_model(model, x_train, y_train, epochs=8, batch_size=128):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
