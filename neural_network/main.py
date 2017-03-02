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
    g = np.dot(X, w1)
    o = np.dot(g, w2)
    return g, o


def bwd_propagation(X, Y_logic, g, o, w1, w2, my_lambda, learning_rate):
    '''
    :param X: features
    :param Y: actual result from data set
    :param g: value of hidden layer
    :param w1: weight between input layer and hidden layer
    :param w2: weight between hidden layer and output layer
    :return: new w1 and w2, after adjustment
    '''
    m = X.shape[0]

    err_out = o - Y_logic
    error_hidden = np.dot(err_out, w2.transpose()) * o * (1-o)

    delta1 = np.dot(X.transpose(), error_hidden) / m
    norm = my_lambda * w1
    norm[0, :] = 0
    delta1 = delta1 + norm

    delta2 = np.dot(g.transpose(), err_out) / m
    norm = my_lambda * w2
    norm[0, :] = 0
    delta2 = delta2 + norm
    new_w1, new_w2 = update_weight(w1, w2, delta1, delta2, learning_rate)
    return new_w1, new_w2


def update_weight(w1, w2, delta1, delta2, learning_rate):
    '''
    update w1 and w2 according to the result of backward propagation
    :param w1: weight between input layer and hidden layer
    :param w2: weight between hidden layer and output layer
    :param delta1: partial derivative for weight between input and hidden layer
    :param delta2: partial derivative for weight between hidden layer and output layer
    :return: ans1:new w1  and  ans2: new w2
    '''
    ans1 = w1 + learning_rate * delta1
    ans2 = w2 + learning_rate * delta2
    return ans1, ans2


def predict(o):
    '''
    decide which number is it to the highest probability
    :param o: value of output layer
    :return: vector of prediction: predic
    '''
    pre = np.argmax(o, 1)
    return pre


def make_logic_matrix(Y, K):
    m = Y.shape[0]
    ans = np.ndarray(shape=(m, K), dtype=int)
    for i in range(0, m):
        ans[i, :] = (Y[i,1]==K).astype(int)   #here is really trick!! Mark it!
    return ans


def evaluate_neural_network(X_test, Y_test, w1, w2):
    g, o = fwd_propagation(X_test, Y_test, w1, w2)
    Y_pred = predict(o)
    m_test = Y_test.size[0]
    correct_num = np.sum((Y_pred==Y_test).astype(int), 0)[0]
    print("%d correct on %d test sets, accuracy=%%%f" %(correct_num, m_test, (correct_num*100)/m_test))



if __name__ == '__main__':
    training_file_path = "/Users/data/data_hand_write_numbers_training.mat"
    testing_file_path = "/Users/data/data_hand_write_numbers_testing.mat"
    iteration = 50
    learning_rate = 0.1
    hidden_node_num = 25
    K = 10
    my_lambda = 1
    '''
    iteration: iterations of feed forward and backward propagation
    learning_rate: learning rate of gradient decent, which is actually been used here
    hidden_node_num: since we have only on hidden layer, it's convenient to store it
    K: the size of cluster
    my_lambda: panalty rate, applied on the params against over fitting
    '''

    X, Y, X_test, Y_test = load_data(training_file_path, testing_file_path)

    Y_logic = make_logic_matrix(Y, K)
    w1 = np.random.random_sample(size=(X.shape[1]+1, hidden_node_num+1))
    w2 = np.random.random_sample(size=(hidden_node_num+1, K))

    for i in range(0, iteration):
        if i != 0:
            w1, w2 = bwd_propagation(X, Y, g, o, w1, w2, my_lambda, learning_rate)
        if i != iteration - 1:
            g, o = fwd_propagation(X, w1, w2)
        evaluate_neural_network(X_test, Y_test, w1, w2)