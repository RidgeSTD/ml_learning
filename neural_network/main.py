from scipy import io as sio
import numpy as np
import scipy as sp

# using a one hidden layer with 25 nodes model, which is recommended by ML Stanford
# the input data has 5000 samples with each 40 features, namely 20x20 pixels



def load_data(training_file_path, testing_file_path):
    print("start loading data...")
    data = sio.loadmat(training_file_path)
    X = data['X']
    Y = data['y']
    X = np.hstack((np.ones(shape=(X.shape[0], 1), dtype=float), X))  # add bias
    data = sio.loadmat(testing_file_path)
    X_test = data['X']
    Y_test = data['y']
    X_test = np.hstack((np.ones(shape=(X_test.shape[0], 1), dtype=float), X_test))  # add bias
    print("load data finish!")
    return X, Y, X_test, Y_test


def sigmoid(M):
    return 1 / (1 + np.exp(-M))


def feed_fwd(X, w1, w2):
    '''
    :param X: features
    :param w1: weight between input layer and hidden layer
    :param w2: weight between hidden layer and output layer
    :return: value of hidden layer:g and value of output layer:o
    '''
    g = sigmoid(np.dot(X, w1))
    o = sigmoid(np.dot(g, w2))
    return g, o


def bwd_propagation(X, Y_logic, g, o, w1, w2, my_lambda):
    '''
    :param X: features
    :param Y: actual result from data set
    :param g: value of hidden layer
    :param w1: weight between input layer and hidden layer
    :param w2: weight between hidden layer and output layer
    :return: partial deravative for every edge in neural network
    '''
    m = X.shape[0]

    err_out = o - Y_logic
    error_hidden = np.dot(err_out, w2.transpose()) * g * (1-g)

    delta1 = np.dot(X.transpose(), error_hidden)
    norm = my_lambda * w1
    norm[0, :] = 0
    delta1 = delta1 + norm
    delta1 = delta1 / m

    delta2 = np.dot(g.transpose(), err_out)
    norm = my_lambda * w2
    norm[0, :] = 0
    delta2 = delta2 + norm
    delta2 = delta2 / m

    return delta1, delta2


def cost_function(nn_weights, X, Y_logic, my_lambda, hidden_layer_size, input_layer_size, K):
    m = Y_logic.shape(0)
    w1 = nn_weights[0:input_layer_size * hidden_layer_size].reshape(input_layer_size, hidden_layer_size)
    w2 = nn_weights[input_layer_size * hidden_layer_size:].reshape(input_layer_size, K)
    g, o = feed_fwd(X, w1, w2)
    tmp = Y_logic * np.log(o) + (1 - Y_logic) * np.log(1 - o)
    w1[0, :] = 0
    w2[0, :] = 0
    J = (sum(sum(w1**2, 0)) + sum(sum(w2**2, 0))) * (my_lambda / 2)
    J = J + sum(sum(tmp, 0))
    J = J / m
    return J


def gradient(nn_weights, X, Y_logic, my_lambda, hidden_layer_size, input_layer_size, K):
    m = Y_logic.shape[0]
    w1 = nn_weights[0:input_layer_size * hidden_layer_size].reshape(input_layer_size, hidden_layer_size)
    w2 = nn_weights[input_layer_size * hidden_layer_size:].reshape(input_layer_size, K)
    g, o = feed_fwd(X, w1, w2)
    delta1, delta2 = bwd_propagation(X, Y_logic, g, o, w1, w2, my_lambda)
    return np.hstack((delta1.reshape(delta1.size), delta2.reshape(delta2.size)))



def update_weight(w1, w2, delta1, delta2, learning_rate):
    '''
    **discarded**
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


def predict(o_p):
    '''
    decide which number is it to the highest probability
    :param o: value of output layer
    :return: vector of prediction: predic
    '''
    pre = np.argmax(o_p, 1)
    return pre


def make_logic_matrix(Y, K):
    m = Y.shape[0]
    ans = np.ndarray(shape=(m, K), dtype=int)
    tmp = np.linspace(1, K, K, dtype=int)
    for i in range(0, m):
        ans[i, :] = (Y[i,0]==tmp).astype(int)   #here is really trick!! Mark it!
    return ans


def evaluate_neural_network(X_test, Y_test, w1, w2):
    g_e, o_e = feed_fwd(X_test, w1, w2)
    Y_pred = predict(o_e)
    m_test = Y_test.shape[0]
    correct_num = np.sum((Y_pred==Y_test).astype(int), 0)[0]
    print("%d correct on %d test sets, accuracy=%f%%" %(correct_num, m_test, (correct_num*100)/m_test))


if __name__ == '__main__':
    training_file_path = "/Users/data/data_hand_write_numbers_training.mat"
    testing_file_path = "/Users/data/data_hand_write_numbers_testing.mat"
    iteration = 5000
    # learning_rate = 0.1
    hidden_layer_size = 26
    input_layer_size = 401
    K = 10
    my_lambda = 3
    epsilon = 0.12
    '''
    iteration: iterations of feed forward and backward propagation
    # learning_rate: learning rate of gradient decent, which is actually been used here
    hidden_node_num: since we have only on hidden layer, it's convenient to store it
    K: the size of cluster
    my_lambda: panalty rate, applied on the params against over fitting
    epsilon: range for randomly initialize w1 and w2
    '''

    X, Y, X_test, Y_test = load_data(training_file_path, testing_file_path)

    Y_logic = make_logic_matrix(Y, K)
    w1 = np.random.random_sample(X.shape[1] * (hidden_layer_size)) * 2 * epsilon - epsilon
    w2 = np.random.random_sample((hidden_layer_size) * K) * 2 * epsilon - epsilon

    nn_weights_opt, J_opt = sp.optimize.fmin_bfgs(f=cost_function, x0=np.hstack(w1, w1), fprime=gradient,
                                                  args=(X, Y_logic, my_lambda, hidden_layer_size, input_layer_size, K),
                                                  full_output=True)
    w1 = nn_weights_opt[0:input_layer_size * hidden_layer_size].reshape(input_layer_size, hidden_layer_size)
    w2 = nn_weights_opt[input_layer_size * hidden_layer_size:].reshape(input_layer_size, K)
    evaluate_neural_network(X_test, Y_test, w1, w2)