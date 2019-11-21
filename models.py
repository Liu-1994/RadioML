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

    output = Dense(num_class, activation='softmax')(x)
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


def crmrn_cnn(num_classes):
    dr = 0.5
    model = keras.models.Sequential()
    in_shp = [2, 128]
    model.add(Reshape(([1] + in_shp), input_shape=in_shp))
    model.add(ZeroPadding2D((0, 2)))
    model.add(Conv2D(256, (1, 3), padding='valid', activation="relu", name="conv1", init='glorot_uniform',
                     data_format="channels_first"))
    model.add(Dropout(dr))
    model.add(ZeroPadding2D((0, 2)))
    model.add(Conv2D(80, (2, 3), padding="valid", activation="relu", name="conv2", init='glorot_uniform',
                     data_format="channels_first"))
    model.add(Dropout(dr))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', init='he_normal', name="dense1"))
    model.add(Dropout(dr))
    model.add(Dense(num_classes, init='he_normal', name="dense2"))
    model.add(Activation('softmax'))
    model.summary()
    return model

if __name__ == '__main__':
    latter_cnn(2)
