import time
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import os
import nn_model
import imageio
from skimage.transform import resize
from sklearn.utils import shuffle

def normalize_it(X):
    print(X.shape)
    v_min = X.min(axis=(2, 3), keepdims=True)
    v_max = X.max(axis=(2, 3), keepdims=True)
    X = (X - v_min)/(v_max - v_min)
    X = np.nan_to_num(X)
    return X


def generate_train_val_test(cropped_dir="cropped"):
    max_seq_length = 29

    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []


    MAX_WIDTH = 100
    MAX_HEIGHT = 100

    t1 = time.time()

    for word in os.listdir(cropped_dir):
        tx1 = time.time()
        word_folder = f"{cropped_dir}/{word}"
        for word_instance in os.listdir(f"{cropped_dir}/{word}"):
            word_instance_folder = f"{word_folder}/{word_instance}"
            sequence = []
            for frame in os.listdir(f"{cropped_dir}/{word}/{word_instance}"):
                image_path = f"{cropped_dir}/{word}/{word_instance}/{frame}"
                output_path = f"{word_instance_folder}/{frame}"
                image = imageio.imread(image_path)
                image = resize(image, (MAX_WIDTH, MAX_HEIGHT))
                image = 255 * image
                # Convert to integer data type pixels.
                image = image.astype(np.uint8)
                sequence.append(image)    

            pad_array = [np.zeros((MAX_WIDTH, MAX_HEIGHT))]
            sequence.extend(pad_array * (max_seq_length - len(sequence)))
            sequence = np.array(sequence)

            if int(word_instance.split("_")[-1]) >= 3:
                X_test.append(sequence)
                y_test.append(word)
            else:
                X_train.append(sequence)
                y_train.append(word)

        tx2 = time.time()
        print(f'Finished reading images for word {word}. Time taken : {tx2 - tx1} secs.')
        
    t2 = time.time()
    print(f"Time taken for creating constant size 3D Tensors from those cropped lip regions : {t2 - t1} secs.")

    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    # print(X_train.shape)
    # print(X_val.shape)
    # print(X_test.shape)
    # print(y_train.shape)
    # print(y_val.shape)
    # print(y_test.shape)


    X_train = normalize_it(X_train)
    # X_val = normalize_it(X_val)
    X_test = normalize_it(X_test)

    
    # dictionary mapping words in cropped_dir to its indexes
    word_to_index = {word: index for index, word in enumerate(os.listdir(cropped_dir))}
    print(word_to_index)
    y_size = len(word_to_index)

    y_train = tf.keras.utils.to_categorical([word_to_index[word] for word in y_train], y_size)
    y_test = tf.keras.utils.to_categorical([word_to_index[word] for word in y_test], y_size)
    # y_val = tf.keras.utils.to_categorical(y_val, y_size)

    X_train, y_train = shuffle(X_train, y_train, random_state=0)
    X_test, y_test = shuffle(X_test, y_test, random_state=0)
    # X_val, y_val = shuffle(X_val, y_val, random_state=0)

    X_train = np.expand_dims(X_train, axis=4)
    # X_val = np.expand_dims(X_val, axis=4)
    X_test = np.expand_dims(X_test, axis=4)
    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)

    train_val_test = np.array([X_train, X_val, X_test, y_train, y_val, y_test])
    with open("train_val_test.npy", "wb") as f:
        np.save(f, train_val_test)

    return train_val_test


def get_train_val_test_split():
    # check if train_val_test.npy exists
    if os.path.exists("train_val_test.npy"):
        with open("train_val_test.npy", "rb") as f:
            train_val_test = np.load(f, allow_pickle=True)
        return train_val_test
    return generate_train_val_test()



def train_nn_model(X_train, y_train, X_val, y_val):
    checkpoint_path = "checkpoints/cp.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)

    model = nn_model.get_cnn_model()

    # with tf.device('/gpu:0'):
    #     t1 = time.time()
    #     # EARLY STOPPING
    #     history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=40, batch_size=4, callbacks=[cp_callback, es_callback])
    #     t2 = time.time()
    #     print(f"Training time : {t2 - t1} secs.")

    t1 = time.time()
    # EARLY STOPPING
    history = model.fit(X_train, y_train, epochs=10, callbacks=[cp_callback, es_callback])
    t2 = time.time()
    print(f"Training time : {t2 - t1} secs.")

    return history


def show_training_graphs(history):
    plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xlim(1, 40)
    # plt.ylim(0, 3)
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


if __name__ == "__main__":
    train_val_test = get_train_val_test_split()
    history = train_nn_model(train_val_test[0], train_val_test[3], train_val_test[1], train_val_test[4])
    show_training_graphs(history)
    print(train_val_test.shape)