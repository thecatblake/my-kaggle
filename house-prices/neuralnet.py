from datetime import datetime

import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from model import make_nn
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.cm as cmx

data = pd.read_csv("./data/train.csv", header=0)
corr = data.select_dtypes(np.number).corr()["SalePrice"]
feature_cols = ["OverallQual", "GrLivArea", "GarageCars"]
pca = PCA(n_components=2)
decomposed = pca.fit_transform(data[feature_cols])

X = tf.convert_to_tensor(decomposed)
Y = tf.convert_to_tensor(data["SalePrice"])

model = make_nn(2)

model.load_weights("./models/neuralnet_500")

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.1)
loss_fn = tf.keras.losses.MeanSquaredError()

model.compile(optimizer, loss_fn)

model.fit(X, Y, epochs=50000, batch_size=256, verbose=1)

model.save("./models/neuralnet_500")

test = pd.read_csv("./data/test.csv", header=0)
test["GarageCars"].fillna(0, inplace=True)
decomposed = pca.fit_transform(test[feature_cols])
predictions = model.predict(decomposed)
pred_data = pd.DataFrame({
    "Id": test["Id"],
    "OverallQual": test["OverallQual"].to_numpy(),
    "GrLivArea": test["GrLivArea"].to_numpy(),
    "GarageCars": test["GarageCars"].to_numpy(),
    "SalePrice": predictions.flatten()
})

decomposed = pca.transform(pred_data[feature_cols])

cm = plt.get_cmap("jet")
cNorm = matplotlib.colors.Normalize(vmin=min(pred_data["SalePrice"]), vmax=max(pred_data["SalePrice"]))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

plt.scatter(decomposed[:, 0], decomposed[:,1], s=5, c=scalarMap.to_rgba(pred_data["SalePrice"]))
plt.show()

pred_data[["Id", "SalePrice"]].to_csv("./nn_predictions_pca_500.csv", index=False)