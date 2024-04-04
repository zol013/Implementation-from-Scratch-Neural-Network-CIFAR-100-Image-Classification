import copy
from neuralnet import *
import util

def modeltrain(model, x_train, y_train, x_valid, y_valid, config):
    """
    Learns the weights (parameters) for our model
    Implements mini-batch SGD to train the model.
    Implements Early Stopping.
    Uses config to set parameters for training like learning rate, momentum, etc.
    args:
        model - an object of the NeuralNetwork class
    returns:
        the trained model
    """
    patience = config['early_stop_epoch']
    batch_size = config['batch_size']
    overfitting = 0
    loss_train = []
    loss_valid = []
    accuracy_train = []
    accuracy_valid = []
    for i in range(config['epochs']):
        # shuffle train set
        length = len(y_train)
        order = np.random.permutation(length)
        loss = 0
        accuracy = 0
        j = 0
        for x, y in util.generate_minibatches((x_train[order],y_train[order]), batch_size):
            loss_cur, accuracy_cur = model(x, y)
            model.backward()
            loss += loss_cur
            accuracy += accuracy_cur
            j += 1

        loss_train.append(loss/j)
        accuracy_train.append(accuracy/j)

        loss, accuracy = model(x_valid, y_valid)
        if loss_valid and loss > loss_valid[-1]:
            overfitting += 1
        loss_valid.append(loss)
        accuracy_valid.append(accuracy)

        if overfitting >= patience:
            break
    util.plots(loss_train, accuracy_train, loss_valid, accuracy_valid, -1)
    return model

#This is the test method
def modelTest(model, X_test, y_test):
    """
    TODO
    Calculates and returns the accuracy & loss on the test set.
    args:
        model - the trained model, an object of the NeuralNetwork class
        X_test - the test set examples
        y_test - the test set targets/labels
    returns:
        test accuracy
        test loss
    """
    loss, accuracy = model(X_test, y_test)
    return loss, accuracy