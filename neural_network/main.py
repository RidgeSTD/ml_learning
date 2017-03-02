from scipy import io as sio
import numpy as np

# using a one hidden layer with 25 nodes model, which is recommended by ML Stanford
# the input data has 5000 samples with each 40 features, namely 20x20 pixels



def load_data(training_file_path, testing_file_path):
    print("start loading data...")
    data = sio.loadmat(training_file_path)
    X = data['X']
    Y = data['y']
    X = np.vstack((np.ones(shape=(1, X.shape[1]), dtype=float), X))  # add bias
    data = sio.loadmat(testing_file_path)
    X_test = data['X']
    Y_test = data['y']
    X_test = np.vstack((np.ones(shape=(1, X_test.shape[1]), dtype=float), X))  # add bias
    print("load data finish!")
    return X, Y, X_test, Y_test


def fwd_propagation(X, w1, w2):
    '''
    :param X: features
    :param w1: weight between input layer and hidden layer
    :param w2: weight between hidden layer and output layer
    :return: value of hidden layer:g and value of output layer:o
    '''
    pass


def bwd_propagation(X, Y, g, w1, w2):
    '''
    :param X: features
    :param Y: actual result from data set
    :param g: value of hidden layer
    :param w1: weight between input layer and hidden layer
    :param w2: weight between hidden layer and output layer
    :return: new w1 and w2
    '''
    pass


def update_weight(w1, w2):
    '''
    update w1 and w2 according to the result of backward propagation
    :param w1: weight between input layer and hidden layer
    :param w2: weight between hidden layer and output layer
    :return: new w1 and w2
    '''
    pass


def predict(o):
    '''
    decide which number is it to the highest probability
    :param o: value of output layer
    :return: vector of prediction: predic
    '''
    pass


if __name__ == '__main__':
    training_file_path = "/Users/data/data_hand_write_numbers_training.mat"
    testing_file_path = "/Users/data/data_hand_write_numbers_testing.mat"
    iteration = 50
    learning_rate = 0.1
    hidden_node_num = 25

    X, Y, X_test, Y_test = load_data(training_file_path, testing_file_path)

    w1 = np.random.random_sample(size=(X.shape[1]+1, hidden_node_num+1))
    w2 = np.random.random_sample(size=(hidden_node_num+1, 10))

    # for i in range(0, iteration):
    #     if i != 0:
    #         bwd_propagation(X, Y, g, w1, w2)
        #fwd_propagation
    pass