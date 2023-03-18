#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @file main.py
# @brief
# @author haogao
# @date 2023/3/18


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import pandas as pd
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
import seaborn as sns
from matplotlib import pyplot as plt


TRAIN_DATA = "./data/train/train_data.csv"
VAL_DATA = "./data/val/val_data.csv"
TEST_DATA = "./data/test/test_data.csv"


def load_data():
    train = pd.read_csv(TRAIN_DATA)
    val = pd.read_csv(VAL_DATA)
    test = pd.read_csv(TEST_DATA)

    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)

    data = dict()
    data["train_X"] = train[:, 0:10]
    data["train_y"] = train[:, 10]
    data["val_X"] = val[:, 0:10]
    data["val_y"] = val[:, 10]
    data["test_X"] = test[:, 0:10]
    data["test_y"] = test[:, 10]

    data["scale"] = scaler
    return data


def build_mlp_network(input_features=None):
    inputs = Input(shape=(input_features,), name="input")
    x = Dense(32, activation='relu', name="hidden")(inputs)
    prediction = Dense(1, activation='linear', name='final')(x)
    model = Model(inputs=inputs, outputs=prediction)
    adam_optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)
    model.compile(optimizer=adam_optimizer, loss='mean_absolute_error')
    return model


def build_dnn_network(input_features=None):
    inputs = Input(shape=(input_features,), name="input")
    x = Dense(32, activation='relu', name="hidden1")(inputs)
    x = Dense(32, activation='relu', name="hidden2")(x)
    x = Dense(32, activation='relu', name="hidden3")(x)
    x = Dense(32, activation='relu', name="hidden4")(x)
    x = Dense(16, activation='relu', name="hidden5")(x)
    prediction = Dense(1, activation='linear', name='final')(x)
    model = Model(inputs=inputs, outputs=prediction)
    adam_optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)
    model.compile(optimizer=adam_optimizer, loss='mean_absolute_error')
    return model


if __name__ == '__main__':
    data = load_data()
    input_features = data["train_X"].shape[1]
    #model = build_mlp_network(input_features)
    model = build_dnn_network(input_features)
    print("Network Structure")
    print(model.summary())
    model.fit(x=data["train_X"],
              y=data["train_y"],
              batch_size=32,
              epochs=200,
              verbose=1,
              validation_data=(data["val_X"], data["val_y"]))

    train_MAE = mean_absolute_error(model.predict(data["train_X"]), data["train_y"])
    val_MAE = mean_absolute_error(model.predict(data["val_X"]), data["val_y"])
    test_MAE = mean_absolute_error(model.predict(data["test_X"]), data["test_y"])
    print("Model Train MAE: " + str(train_MAE))
    print("Model Val MAE: " + str(val_MAE))
    print("Model Test MAE: " + str(test_MAE))

    model.save("regression_model.h5")

    plt.title("Predicted Distribution vs. Actual")
    y_hat = model.predict(data["test_X"])
    sns.distplot(y_hat.flatten(), label="y_hat")
    sns.distplot(data["test_y"], label="y_true")
    plt.legend()
    plt.savefig("pred_dist.jpg")
