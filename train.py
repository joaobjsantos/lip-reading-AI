import time
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import os
import nn_model
import imageio
from skimage.transform import resize
from sklearn.utils import shuffle
from split_video_into_frames import get_word_counts_from_file

def normalize_it(X):
    """
    Normalizes the input array `X` by scaling its values between 0 and 1.

    Parameters:
    - X (ndarray): The input array to be normalized.

    Returns:
    - ndarray: The normalized array.
    """
    print(X.shape)
    v_min = X.min(axis=(2, 3), keepdims=True)
    v_max = X.max(axis=(2, 3), keepdims=True)
    X = (X - v_min)/(v_max - v_min)
    X = np.nan_to_num(X)
    return X


def generate_train_val_test(cropped_dir="cropped", word_counts=None, train_val_test_split=[0.6, 0.2, 0.2], max_word_count=40):
    """
    Generate training, validation, and test data from cropped lip images.

    Parameters:
    - `cropped_dir` (str): The directory containing the cropped lip images. Default is "cropped".
    - `word_counts` (dict): A dictionary mapping word classes to their counts. Default is None.
    - `train_val_test_split` (list): A list of three floats representing the train, validation, and test split ratios. Default is [0.6, 0.2, 0.2].
    - `max_word_count` (int): The maximum number of word instances to consider for each word class. Default is 40.

    Returns:
    - `train_val_test` (ndarray): A NumPy array containing the training, validation, and test data.
    
    The function generates training, validation, and test data from cropped lip images stored in the `cropped_dir` directory. It uses the `word_counts` parameter to determine the number of instances for each word class. The function splits the data according to the `train_val_test_split` ratios and pads the sequences to a maximum length of `max_seq_length`. The data is then normalized and converted to constant size 3D tensors. Finally, the data is shuffled and saved to a file named "train_val_test.npy".

    Note: The function assumes that the cropped lip images are stored in the following directory structure: `cropped_dir/word/instance/frame`.
    """
    max_seq_length = 29

    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []


    MAX_WIDTH = 100
    MAX_HEIGHT = 100


    if word_counts is None:
        word_counts = get_word_counts_from_file()


    frequent_words = []

    t1 = time.time()

    for word in os.listdir(cropped_dir):
        tx1 = time.time()
        word_folder = f"{cropped_dir}/{word}"

        word_count = len(os.listdir(word_folder))
        print(f"{word} count = {word_count}, Min accepted = {1/min(train_val_test_split)}")
        if word_count < 1/min(train_val_test_split):
            continue

        frequent_words.append(word)

        if word_count > max_word_count:
            word_count = max_word_count
            

        word_i = 1

        for word_instance in os.listdir(word_folder):
            sequence = []
            for frame in os.listdir(f"{cropped_dir}/{word}/{word_instance}"):
                image_path = f"{cropped_dir}/{word}/{word_instance}/{frame}"
                image = imageio.imread(image_path)
                image = resize(image, (MAX_WIDTH, MAX_HEIGHT))
                image = 255 * image
                # Convert to integer data type pixels.
                image = image.astype(np.uint8)
                sequence.append(image)    

            if len(sequence) > max_seq_length:
                sequence = sequence[:max_seq_length]

            pad_array = [np.zeros((MAX_WIDTH, MAX_HEIGHT))]
            sequence.extend(pad_array * (max_seq_length - len(sequence)))
            sequence = np.array(sequence)

            
            if word_i < train_val_test_split[0] * word_count:
                X_train.append(sequence)
                y_train.append(word)
            elif word_i < train_val_test_split[0] * word_count + train_val_test_split[1] * word_count:
                # print(f"val {sequence.shape} {np.array(X_val).shape}")
                X_val.append(sequence)
                y_val.append(word)
            else:
                X_test.append(sequence)
                y_test.append(word)

            word_i += 1

            if word_i >= max_word_count:
                break


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

    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_val.shape)
    print(y_test.shape)


    X_train = normalize_it(X_train)
    X_val = normalize_it(X_val)
    X_test = normalize_it(X_test)

    
    # dictionary mapping words in cropped_dir to its indexes
    word_to_index = {word: index for index, word in enumerate(frequent_words)}
    print(word_to_index)
    y_size = len(word_to_index)

    y_train = tf.keras.utils.to_categorical([word_to_index[word] for word in y_train], y_size)
    y_val = tf.keras.utils.to_categorical([word_to_index[word] for word in y_val], y_size)
    y_test = tf.keras.utils.to_categorical([word_to_index[word] for word in y_test], y_size)
    

    X_train, y_train = shuffle(X_train, y_train, random_state=0)
    X_val, y_val = shuffle(X_val, y_val, random_state=0)
    X_test, y_test = shuffle(X_test, y_test, random_state=0)
    

    X_train = np.expand_dims(X_train, axis=4)
    X_val = np.expand_dims(X_val, axis=4)
    X_test = np.expand_dims(X_test, axis=4)

    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)
    print(np.array(y_train).shape)
    print(np.array(y_val).shape)
    print(np.array(y_test).shape)

    train_val_test = np.array([X_train, X_val, X_test, y_train, y_val, y_test, frequent_words])
    with open("train_val_test.npy", "wb") as f:
        np.save(f, train_val_test)

    return train_val_test


def get_train_val_test_split(cropped_dir="cropped", max_word_count=40):
    # check if train_val_test.npy exists
    if os.path.exists("train_val_test.npy"):
        with open("train_val_test.npy", "rb") as f:
            train_val_test = np.load(f, allow_pickle=True)
        return train_val_test
    return generate_train_val_test(cropped_dir=cropped_dir, max_word_count=max_word_count)



def train_nn_model(X_train, y_train, X_val, y_val, gpu=False, model=None, model_name="3_layer_CNN"):
    """
	Train the neural network model.

	Parameters:
	- X_train: The training data.
	- y_train: The target labels for the training data.
	- X_val: The validation data.
	- y_val: The target labels for the validation data.
	- gpu: Whether to use GPU for training. Default is False.
	- model: The pre-trained model to use. Default is None.
	- model_name: The name of the model architecture to use. Default is "3_layer_CNN".

	Returns:
	- history: The training history.
	- model: The trained model.
    """
    
    if gpu:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], enable=True)
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)])
        print(gpus)

    checkpoint_path = "./checkpoints/cp.ckpt"
    #cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)

    if model is None:
        if model_name == "3_layer_CNN":
            model = nn_model.get_cnn_model(num_classes=y_train.shape[1])
        elif model_name == "4_layer_CNN":
            model = nn_model.get_4_layer_cnn_model(num_classes=y_train.shape[1])
        else:
            model = nn_model.get_cnn_lstm_model(num_classes=y_train.shape[1])

    if gpu:
        with tf.device('/gpu:0'):
            t1 = time.time()
            # EARLY STOPPING
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=4, callbacks=[es_callback])
            t2 = time.time()
            print(f"Training time : {t2 - t1} secs.")
    else:
        t1 = time.time()
        # EARLY STOPPING
        history = model.fit(X_train, y_train, epochs=10, callbacks=[es_callback])
        t2 = time.time()
        print(f"Training time : {t2 - t1} secs.")

    return history, model


def show_training_graphs(history):
    """
    Generates and displays training graphs based on the history of the trained model.

    Args:
        history (object): The history object that contains the training history.

    Returns:
        None
    """
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


def split_only(cropped_dir="cropped"):
    """
    Generate the train, validation, and test split for the given directory.

    Args:
        cropped_dir (str, optional): The directory path where the cropped images are stored. Defaults to "cropped".

    Returns:
        dict: A dictionary containing the train, validation, and test split.
    """
    train_val_test = get_train_val_test_split(cropped_dir=cropped_dir)
    return train_val_test

def split_and_train(cropped_dir="cropped", gpu=False, model_name="3_layer_CNN", max_word_count=40):
    """
    This function splits the data into training, validation, and testing sets and trains a neural network model.

    :param cropped_dir: (str) The directory containing the cropped images. Default is "cropped".
    :param gpu: (bool) Whether to use GPU for training. Default is False.
    :param model_name: (str) The name of the model to be trained. Default is "3_layer_CNN".
    :param max_word_count: (int) The maximum instance count for each word. Default is 40.
    
    :return: (tf.keras.Model) The trained neural network model.
    """
    train_val_test = get_train_val_test_split(cropped_dir=cropped_dir, max_word_count=max_word_count)
    history, model = train_nn_model(train_val_test[0], train_val_test[3], train_val_test[1], train_val_test[4], gpu, model_name=model_name)
    show_training_graphs(history)
    print(train_val_test.shape)
    
    return model

if __name__ == "__main__":
    # split_and_train()
    split_only()
	