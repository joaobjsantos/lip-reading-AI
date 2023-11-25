import time
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import os
import nn_model

def normalize_it(X):
    """
    Normalize the input array X along specified axes by scaling its values between 0 and 1.

    Parameters:
        X (ndarray): Input array to be normalized.

    Returns:
        ndarray: Normalized array with values scaled between 0 and 1.
    """
    v_min = X.min(axis=(2, 3), keepdims=True)
    v_max = X.max(axis=(2, 3), keepdims=True)
    X = (X - v_min)/(v_max - v_min)
    X = np.nan_to_num(X)
    return X


def generate_train_val_test_miraclvc1():
    """
    Generates train, validation, and test data for MIRACL-VC1 dataset.

    Parameters:
        None

    Returns:
        numpy.ndarray: An array containing train, validation, and test data and labels in the following order:
            - X_train: numpy.ndarray, shape (num_train_samples, max_seq_length, MAX_WIDTH, MAX_HEIGHT), containing the train data
            - X_val: numpy.ndarray, shape (num_val_samples, max_seq_length, MAX_WIDTH, MAX_HEIGHT), containing the validation data
            - X_test: numpy.ndarray, shape (num_test_samples, max_seq_length, MAX_WIDTH, MAX_HEIGHT), containing the test data
            - y_train: numpy.ndarray, shape (num_train_samples,), containing the train labels
            - y_val: numpy.ndarray, shape (num_val_samples,), containing the validation labels
            - y_test: numpy.ndarray, shape (num_test_samples,), containing the test labels
    """
    max_seq_length = 22

    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []


    MAX_WIDTH = 100
    MAX_HEIGHT = 100

    t1 = time.time()
    # UNSEEN_VALIDATION_SPLIT = ['F07', 'M02']
    # UNSEEN_TEST_SPLIT = ['F04']
    UNSEEN_VALIDATION_SPLIT = ['F07']
    UNSEEN_TEST_SPLIT = ['F08']


    directory = BASE_DIR + "cropped"

    for person_id in people:
        tx1 = time.time()
        for data_type in data_types:
            for word_index, word in enumerate(folder_enum):
                # print(f"Word : '{words[word_index]}'")
                for iteration in instances:
                    path = os.path.join(directory, person_id, data_type, word, iteration)
                    filelist = sorted(os.listdir(path + SLASH_TYPE))
                    sequence = [] 
                    for img_name in filelist:
                        if img_name.startswith('color'):
                            image = imageio.imread(path + SLASH_TYPE + img_name)
                            image = resize(image, (MAX_WIDTH, MAX_HEIGHT))
                            image = 255 * image
                            # Convert to integer data type pixels.
                            image = image.astype(np.uint8)
                            sequence.append(image)                        
                    pad_array = [np.zeros((MAX_WIDTH, MAX_HEIGHT))]                            
                    sequence.extend(pad_array * (max_seq_length - len(sequence)))
                    sequence = np.array(sequence)
                    if person_id in UNSEEN_TEST_SPLIT:
                        X_test.append(sequence)
                        y_test.append(word_index)
                    elif person_id in UNSEEN_VALIDATION_SPLIT:
                        X_val.append(sequence)
                        y_val.append(word_index)
                    else:
                        X_train.append(sequence)
                        y_train.append(word_index)    
        tx2 = time.time()
        print(f'Finished reading images for person {person_id}. Time taken : {tx2 - tx1} secs.')
        
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
    X_val = normalize_it(X_val)
    X_test = normalize_it(X_test)

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
    return generate_train_val_test_miraclvc1()



def train_nn_model(X_train, y_train, X_val, y_val):
    checkpoint_path = "checkpoints/cp.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)

    model = nn_model.get_nn_model()

    with tf.device('/gpu:0'):
        t1 = time.time()
        # EARLY STOPPING
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=40, batch_size=4, callbacks=[cp_callback, es_callback])
        t2 = time.time()
        print(f"Training time : {t2 - t1} secs.")

    return history


def show_training_graphs(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
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