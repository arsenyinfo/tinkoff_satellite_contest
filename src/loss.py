from functools import partial

from keras.losses import binary_crossentropy
from keras import backend as K


def iou(y_true, y_pred):
    sum_ = partial(K.sum, axis=(1, 2))
    iou_ = ((sum_(y_true * y_pred) + K.epsilon()) / (sum_(y_true) + sum_(y_pred) - sum_(y_true * y_pred) + K.epsilon()))
    return K.mean(iou_)


def bce_log_iou(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(iou(y_true, y_pred))
