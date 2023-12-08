import keras


def make_nn(n_inp):
    return keras.Sequential([
        keras.layers.Dense(20, input_shape=(n_inp,), activation='relu'),
        keras.layers.Dense(20, activation='relu'),
        keras.layers.Dense(1)
    ])