import keras
from keras.layers import *
from keras.models import Model
from keras.activations import *


def dr_cnn(num_class):
    input = Input((2, 128, 1))

    x = Conv2D(128, (2, 8), padding='valid')(input)
    x = PReLU()(x)
    x = Dropout(0.5)(x)

    x = Conv2D(64, (1, 16), padding='valid')(x)
    x = PReLU()(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(128)(x)
    x = PReLU()(x)
    x = Dropout(0.5)(x)

    x = Dense(64)(x)
    x = PReLU()(x)
    x = Dropout(0.5)(x)

    x = Dense(32)(x)
    x = PReLU()(x)
    x = Dropout(0.5)(x)

    output = Dense(num_class, activation='softmax', kernel_regularizer=regularizers.l2())(x)
    model = Model(inputs=input, outputs=output)
    model.summary()
    return model


def latter_cnn(num_class):
    input = Input((128, 128, 3))

    x = Conv2D(128, (5, 5), padding='valid')(input)
    x = PReLU()(x)
    x = AveragePooling2D((2, 2), strides=2)(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = PReLU()(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = PReLU()(x)
    x = AveragePooling2D((2, 2), strides=2)(x)

    x = Conv2D(32, (3, 3), padding='same')(x)
    x = PReLU()(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = PReLU()(x)
    x = AveragePooling2D((2, 2), strides=2)(x)

    x = Flatten()(x)
    x = Dense(1024, kernel_regularizer=regularizers.l2())(x)
    x = PReLU()(x)
    x = Dropout(0.5)(x)

    x = Dense(512, kernel_regularizer=regularizers.l2())(x)
    x = PReLU()(x)
    x = Dropout(0.5)(x)

    output = Dense(num_class, activation='softmax', kernel_regularizer=regularizers.l2())(x)
    model = Model(inputs=input, outputs=output)
    model.summary()
    return model