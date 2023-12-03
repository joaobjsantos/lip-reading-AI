from keras.layers import Conv3D, MaxPooling3D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.layers import Activation, ZeroPadding3D, TimeDistributed, LSTM, GRU, Reshape
from keras.layers import BatchNormalization

def get_simple_cnn_model(image_size=29, num_classes=7):
    model = Sequential()

    # 1st layer group
    model.add(Conv3D(64, (3, 3, 3), strides = 1, input_shape=(image_size, 100, 100, 1), activation='relu', padding='valid'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2))

    model.add((Flatten()))

    # # FC layers group
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.5))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    # model.summary()

    return model

def get_cnn_model(image_size=29, num_classes=7):
    """
    Returns a neural network model.
    
    Returns:
        model (Sequential): The neural network model.

    """
    model = Sequential()

    # 1st layer group
    model.add(Conv3D(64, (3, 3, 3), strides = 1, input_shape=(image_size, 100, 100, 1), activation='relu', padding='valid'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2))

    model.add(Conv3D(128, (3, 3, 3), activation='relu', strides=1))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2))

    model.add(Conv3D(256, (2, 2, 2), activation='relu', strides=1))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2))

    model.add((Flatten()))

    # # FC layers group
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(.5))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    # model.summary()

    return model


def get_4_layer_cnn_model(image_size=29, num_classes=7):
    model = Sequential()

    # 1st layer group
    model.add(Conv3D(64, (3, 3, 3), strides = 1, input_shape=(image_size, 100, 100, 1), activation='relu', padding='valid'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2))

    model.add(Conv3D(128, (3, 3, 3), activation='relu', strides=1))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2))

    model.add(Conv3D(256, (2, 2, 2), activation='relu', strides=1))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2))

    model.add(Conv3D(512, (1, 1, 1), activation='relu', strides=1))
    model.add(MaxPooling3D(pool_size=(1, 1, 1), strides=2))

    model.add((Flatten()))

    # # FC layers group
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(.5))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])

    return model


def get_cnn_lstm_model(image_size=29, num_classes=7):
    model = Sequential()

    # 1st layer group
    model.add(Conv3D(32, (3, 3, 3), strides = 1, input_shape=(image_size, 100, 100, 1), activation='relu', padding='valid'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2))

    model.add(Conv3D(64, (3, 3, 3), activation='relu', strides=1))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2))

    model.add(Conv3D(128, (3, 3, 3), activation='relu', strides=1))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2))

    shape = model.output_shape
    model.add(Reshape((shape[-1],shape[1]*shape[2]*shape[3])))

    # LSTMS - Recurrent Network Layer
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(.5))

    model.add((Flatten()))

    # # FC layers group
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(.5))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    return model


if __name__ == "__main__":
    model = get_cnn_lstm_model()