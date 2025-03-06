import keras


def make_nn(n_inp):
    return keras.Sequential([
        keras.layers.Dense(500, input_shape=(n_inp,), activation='relu'),
        keras.layers.Dense(1)
    ])