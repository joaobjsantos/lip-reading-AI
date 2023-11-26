import os
import numpy as np
import nn_model
from train import get_train_val_test_split

def test_miraclvc1(model, X_test, y_test):
    """
    Calculate the accuracy of a MIRACL-VC1 dataset model on unseen test data.

    Args:
        model (MIRACL-VC1): The trained MIRACL-VC1 dataset model.
        X_test (np.ndarray): The test data features.
        y_test (np.ndarray): The test data labels.

    Returns:
        None
    """
    words = ['Begin', 'Choose', 'Connection', 'Navigation', 'Next', 'Previous', 'Start', 'Stop', 'Hello', 'Web']  
    ypred = model.predict(X_test)
    predicted_words = [words[i] for i in np.argmax(ypred, axis=1)]
    actual_words = [words[i] for i in y_test] 

    correct = 0
    for p, a in zip(predicted_words, actual_words):
        if p == a:
            correct += 1
        print(f"Predicted : {p} \t Actual : {a}")

    accuracy = correct/len(actual_words)
    print(f"Accuracy = {accuracy} on completely unseen data")


def test_model(model, X_test, y_test, cropped_dir="cropped"):
    words = os.listdir(cropped_dir)
    ypred = model.predict(X_test)
    predicted_words = [words[i] for i in np.argmax(ypred, axis=1)]
    actual_words = [words[i] for i in np.argmax(y_test, axis=1)] 

    correct = 0
    for p, a in zip(predicted_words, actual_words):
        if p == a:
            correct += 1
        print(f"Predicted : {p} \t Actual : {a}")

    accuracy = correct/len(actual_words)
    print(f"Accuracy = {accuracy} on completely unseen data")


def test_cnn_model(model=None):
    if model is None:
        model = nn_model.get_cnn_model()
        model.load_weights("checkpoints/cp.ckpt")
    train_val_test = get_train_val_test_split()
    print(train_val_test[5])
    test_model(model, train_val_test[2], train_val_test[5])

if __name__ == "__main__":
    test_cnn_model()

