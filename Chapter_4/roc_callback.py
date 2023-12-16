#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @file roc_callback.py
# @brief
# @author haogao
# @date 2023/4/10


from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score


class RocAUCScore(Callback):
    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        super().__init__()

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\n  *** ROC AUC Score: %s - roc-auc_val: %s ***' % (str(roc), str(roc_val)))
        return