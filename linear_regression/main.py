import numpy as np
from scipy import optimize


iter_default = 1500


def get_data(datafilePath):
    file = open(datafilePath, "r")
    lines = list()
    while True:
        line = file.readline()
        if len(line) == 0:
            break
        lines.append(line)
    m = len(lines)
    if m == 0:
        return 0, 0, None, None
    n = len(lines[0].split(','))
    x = np.ones(shape=(m, n))
    y = np.zeros(shape=(m, 1))
    for i in range(0, m):
        line = lines[i].split(',')
        y[i] = float(line[n-1])
        for j in range(1, n):
            x[i, j] = float(line[j-1])

    # normalization
    mu = np.mean(x, 0)
    mu[0] = 0
    scale = np.max(x, 0) - np.min(x, 0)
    scale[0] = 1
    for i in range(0, m):
        x[i, :] = (x[i, :] - mu) / scale
    # end of normalization

    #  extracting testing data
    test_size = int(m / 5)
    x_test = x[0:test_size, :]
    y_test = y[0:test_size, :]
    x = x[test_size:, :]
    y = y[test_size:, :]
    m -= test_size

    return m, n, x, y, x_test, y_test


def cost_func(theta, x, y, m):
    # a bug accured solved: theata is a ndarray after numpy.chararray.flatten, whose shape/demension information is lost
    # therefore we need to reconstruct theta
    tmp = np.dot(x, theta.reshape(x.shape[1], 1)) - y
    return np.dot(tmp.transpose(), tmp) / m / 2


# def cost_func_prime(theta):
#     global x
#     global y
#     global m
#     tmp = theta * x.transpose() - y


def main_loop(input_filepath):
    global iter_default
    print('parsing training and testing data...')
    m, n, x, y, x_test, y_test = get_data(input_filepath)
    print("finish parsing data, learning...")
    theta_opt = optimize.fmin_bfgs(cost_func, np.zeros((n, 1)), args=(x, y, m))
    print('finish learning, get optimized theta with:')
    print(theta_opt)
    print('\nprinting estimation(s) on %d testing set' % len(y_test))
    for i in range(0, len(x_test)):
        y_est = np.dot(x_test[i, :], theta_opt.transpose())
        print("%d:\ty_est=%f\t\ty_real=%f" % (i, y_est, y_test[i]))



if __name__ == '__main__':
    params_path = input("input params file path")
    params_file = open(params_path)
    params_lines = params_file.readlines()
    data_set_num = len(params_lines)
    counter = 0
    for line in params_lines:
        main_loop(line.strip())
        counter += 1
        print("%d of %d dataset(s) finished!" % (counter, data_set_num))