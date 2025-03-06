# This script needs these libraries to be installed:
#   numpy, xgboost

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import wandb
from wandb.integration.xgboost import WandbCallback

import numpy as np
import xgboost as xgb
import pandas as pd

# setup parameters for xgboost
param = {
    "objective" : "binary:logistic",
    "eta" : 0.1,
    "max_depth": 6,
    "nthread" : 4
}

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="kaggle",

    # track hyperparameters and run metadata
    config=param
)

# download data from wandb Artifacts and prep data
data = pd.read_csv("train.csv", header=0, index_col=0)

train_x, test_x, train_y, test_y = train_test_split(data.drop("rainfall", axis=1), data["rainfall"], test_size=0.2, random_state=42)       

model = xgb.XGBClassifier(
    objective="binary:logistic",
    eta=0.1,
    max_depth=6,
    nthread=4
)

model.fit(train_x, train_y)

predictions = model.predict(test_x)
score = accuracy_score(test_y, predictions)

wandb.log({
    "accuracy": score
})

wandb.finish()
