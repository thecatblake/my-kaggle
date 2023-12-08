from datetime import datetime

import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from model import make_nn

data = pd.read_csv("./data/train.csv", header=0)
feature_cols = ["OverallQual", "GrLivArea"]
X = tf.convert_to_tensor(data[feature_cols])
Y = tf.convert_to_tensor(data["SalePrice"])

model = make_nn(len(feature_cols))

#model.load_weights("./models/neuralnet")

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.1)
loss_fn = tf.keras.losses.MeanSquaredError()

model.compile(optimizer, loss_fn)

model.fit(X, Y, epochs=500000, batch_size=256, verbose=1)

model.save("./models/neuralnet")
