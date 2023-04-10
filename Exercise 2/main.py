#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @file main
# @brief
# @author haogao
# @date 2023/4/9


import os
from sklearn.preprocessing import StandardScaler
import pandas as pd
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from roc_callback import RocAUCScore


TRAIN_DATA = "./data/train/train_data.csv"
VAL_DATA = "./data/val/val_data.csv"
TEST_DATA = "./data/test/test_data.csv"


def load_data():
    train = pd.read_csv(TRAIN_DATA)
    val = pd.read_csv(VAL_DATA)
    test = pd.read_csv(TEST_DATA)

    data = dict()
    data["train_y"] = train.pop('y')
    data["val_y"] = val.pop('y')
    data["test_y"] = test.pop('y')

    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)

    data["train_X"] = train
    data["val_X"] = val
    data["test_X"] = test

    data["scaler"] = scaler
    return data


def build_dnn_network(input_features=None):
    inputs = Input(shape=(input_features,), name="input")
    x = Dense(128, activation='relu', name="hidden1")(inputs)
    x = Dense(64, activation='relu', name="hidden2")(x)
    x = Dense(64, activation='relu', name="hidden3")(x)
    x = Dense(32, activation='relu', name="hidden4")(x)
    x = Dense(16, activation='relu', name="hidden5")(x)
    prediction = Dense(1, activation='sigmoid', name='final')(x)
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
    return model


def create_callbacks(data):
    tensorboard_callback = TensorBoard(log_dir=os.path.join(os.getcwd(), "tb_log", "5h_adam_20epochs"), histogram_freq=1, batch_size=32,
                                       write_graph=True, write_grads=False)
    checkpoint_callback = ModelCheckpoint(filepath="./model-weights.{epoch:02d}-{val_accuracy:.6f}.hdf5", monitor='val_accuracy',
                                          verbose=1, save_best_only=True)
    roc_auc_callback = RocAUCScore(training_data=(data["train_X"], data["train_y"]),
                                   validation_data=(data["val_X"], data["val_y"]))
    return [tensorboard_callback, checkpoint_callback, roc_auc_callback]


if __name__ == '__main__':
    data = load_data()
    callbacks = create_callbacks(data)
    input_features = data["train_X"].shape[1]
    model = build_dnn_network(input_features)
    model.fit(x=data["train_X"],
              y=data["train_y"],
              batch_size=32,
              epochs=20,
              verbose=1,
              validation_data=(data["val_X"], data["val_y"]),
              callbacks=callbacks)

