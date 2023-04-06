"""
Classification of CVEP EEG signals based on the EEG2Code neural network [1]. Other implementations of neural networks
with the same purpose are presented.

Reference:

    [1] - Nagel, S., & Spüler, M. (2019).
          World’s fastest brain-computer interface: combining EEG2Code with deep learning.
          PloS one, 14(9), e0221909.

Authors: Ludovic Darmet
Mail: Ludovic.DARMET@isae-supaero.fr
"""

from __future__ import division
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    InputLayer,
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    Permute,
    Flatten,
    Dense,
    BatchNormalization,
    Dropout,
    LeakyReLU,
    Activation,
    SeparableConv2D,
    DepthwiseConv2D,
    SpatialDropout2D,
    Softmax,
    Add,
    GlobalAveragePooling2D,
    concatenate
)
from tensorflow_addons.layers import GELU, Sparsemax


np.random.seed(seed=42)
def EEGnet_patchembeddingdilation(windows_size, n_channel_input):
    dropoutRate = 0.5
    leak = 0.3
    dropoutType = "Dropout"
    poolType = "max"
    factor_time = 2
    act = "GELU"
    out = "softmax"
    patch_size = 3

    if dropoutType == "SpatialDropout2D":
        dropoutType = SpatialDropout2D
    elif dropoutType == "Dropout":
        dropoutType = Dropout

    if poolType == "avg":
        poolType = AveragePooling2D
    elif poolType == "max":
        poolType = MaxPooling2D

    if act == "GELU":
        act = GELU
        act = LeakyReLU

    if out == "sparsemax":
        out = Sparsemax()
    else:
        out = Softmax()

    # construct sequential model
    model = Sequential()

    model.add(InputLayer(input_shape=(n_channel_input, windows_size, 1)))

    # Temporal embedding
    model.add(
        Conv2D(
            filters=15,
            data_format="channels_last",
            kernel_size=(1, patch_size),
            strides=(1, patch_size),
        )
    )
    model.add(BatchNormalization(axis=1, scale=True, center=False))
    model.add(act())

    # Spatial fitering
    model.add(
        Conv2D(
            int(16 * factor_time),
            kernel_size=(n_channel_input, 1),
            data_format="channels_last",
            padding="valid",
            strides=(1, 1),
            kernel_initializer="he_uniform",
            activation=None,
        )
    )
    model.add(BatchNormalization(axis=1, scale=True, center=False))
    # model.add(act())
    # model.add(poolType(pool_size=(2, 1),data_format = 'channels_first',  strides=(2, 1),padding='same'))
    model.add(dropoutType(dropoutRate))

    # Temporal filtering
    model.add(
        Conv2D(
            int(16 * factor_time),
            kernel_size=(1, 20),
            data_format="channels_last",
            padding="same",
            dilation_rate=(1, 2),
            kernel_initializer="he_uniform",
            activation=None,
        )
    )
    model.add(BatchNormalization(axis=1, scale=True, center=False))
    model.add(act())
    model.add(poolType(pool_size=(2, 2), strides=(2, 2), data_format='channels_last', padding='same'))
    model.add(dropoutType(dropoutRate))
    # 2D convo
    model.add(
        Conv2D(
            int(16 * factor_time),
            kernel_size=(5, 5),
            data_format="channels_last",
            padding="same",
            dilation_rate=(2, 2),
            kernel_initializer="he_uniform",
            activation=None,
        )
    )
    model.add(BatchNormalization(axis=1, scale=True, center=False))
    model.add(act())
    model.add(Permute((1, 3, 2)))
    model.add(poolType(pool_size=(2, 2), padding='same'))
    # model.add(dropoutType(dropoutRate))
    # layer4
    model.add(Flatten())
    model.add(Dense(int(128), activation=None))
    model.add(LeakyReLU(alpha=leak))
    # model.add(Dropout(0.5))
    # # layer5
    # model.add(Dense(int(64*factor_time), activation=None))
    # model.add(LeakyReLU(alpha=0.3))
    # model.add(Dropout(0.5))
    # layer6
    model.add(Dense(2, name="preds", activation=None))
    model.add(out)
    # print(model.summary())
    return model


if __name__ == "__main__":
    from tensorflow.keras.utils import plot_model

    clf = EEGnet_patchembeddingdepthwise(125, 30)
    clf.summary()
    # plot_model(clf, to_file='models_archi/EEGnet_patchembeddingdilation.png', show_shapes=True, show_layer_names=True)
