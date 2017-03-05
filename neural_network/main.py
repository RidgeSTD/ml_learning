from scipy import io as sio
import numpy as np
from scipy import optimize

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
    input_hidden_layer = np.dot(X, w1)
    output_hidden_layer = sigmoid(input_hidden_layer)
    input_output_layer = np.dot(output_hidden_layer, w2)
    output_output_layer = sigmoid(input_output_layer)
    return input_hidden_layer, output_hidden_layer, input_output_layer, output_output_layer



def cost_function(nn_weights, X, Y_logic, my_lambda, hidden_layer_size, input_layer_size, K):
    m = Y_logic.shape[0]
    w1 = nn_weights[0:input_layer_size * hidden_layer_size].reshape(input_layer_size, hidden_layer_size)
    w2 = nn_weights[input_layer_size * hidden_layer_size:].reshape(hidden_layer_size, K)
    input_hidden_layer, output_hidden_layer, input_output_layer, output_output_layer = feed_fwd(X, w1, w2)
    tmp = Y_logic * np.log(output_output_layer) + (1 - Y_logic) * np.log(1 - output_output_layer)
    w1[0, :] = 0
    w2[0, :] = 0
    J = (sum(sum(w1**2, 0)) + sum(sum(w2**2, 0))) * (my_lambda / 2)
    J = J + sum(sum(tmp, 0))
    J = J / m
    return J


def gradient(nn_weights, X, Y_logic, my_lambda, hidden_layer_size, input_layer_size, K):
    m = Y_logic.shape[0]
    w1 = nn_weights[0:input_layer_size * hidden_layer_size].reshape(input_layer_size, hidden_layer_size)
    w2 = nn_weights[input_layer_size * hidden_layer_size:].reshape(hidden_layer_size, K)

    input_hidden_layer, output_hidden_layer, input_output_layer, output_output_layer = feed_fwd(X, w1, w2)

    # backward propagation
    err_out = output_output_layer - Y_logic
    error_hidden = np.dot(err_out, w2.transpose()) * input_hidden_layer * (1 - input_hidden_layer)

    delta_input = np.dot(X.transpose(), error_hidden)
    norm = my_lambda * w1
    norm[0, :] = 0
    delta_input = delta_input + norm
    delta_input = delta_input / m

    delta_hidden = np.dot(output_hidden_layer.transpose(), err_out)
    norm = my_lambda * w2
    norm[0, :] = 0
    delta_hidden = delta_hidden + norm
    delta_hidden = delta_hidden / m

    return np.hstack((delta_input.reshape(delta_input.size), delta_hidden.reshape(delta_hidden.size)))



# def update_weight(w1, w2, delta1, delta2, learning_rate):
#     '''
#     **discarded**
#     update w1 and w2 according to the result of backward propagation
#     :param w1: weight between input layer and hidden layer
#     :param w2: weight between hidden layer and output layer
#     :param delta1: partial derivative for weight between input and hidden layer
#     :param delta2: partial derivative for weight between hidden layer and output layer
#     :return: ans1:new w1  and  ans2: new w2
#     '''
#     ans1 = w1 + learning_rate * delta1
#     ans2 = w2 + learning_rate * delta2
#     return ans1, ans2


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
    # TODO
    input_hidden_layer, output_hidden_layer, input_output_layer, output_output_layer = feed_fwd(X_test, w1, w2)
    Y_pred = predict(output_output_layer)
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
    w1 = np.random.random_sample(input_layer_size * hidden_layer_size) * 2 * epsilon - epsilon
    w2 = np.random.random_sample(hidden_layer_size * K) * 2 * epsilon - epsilon

    [x_opt, f_opt,gopt, Bopt, func_calls, grad_calls, warn_flag] \
        = optimize.fmin_bfgs(f=cost_function, x0=np.hstack((w1, w2)), fprime=gradient,
                             args=(X, Y_logic, my_lambda, hidden_layer_size, input_layer_size, K),full_output=True)
    w1 = x_opt[0:input_layer_size * hidden_layer_size].reshape(input_layer_size, hidden_layer_size)
    w2 = x_opt[input_layer_size * hidden_layer_size:].reshape(hidden_layer_size, K)
    evaluate_neural_network(X_test, Y_test, w1, w2)