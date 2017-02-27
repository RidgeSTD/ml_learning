import numpy as np


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
        y[i] = line[n]
        for j in range(1, n):
            x[i, j] = float(line[j-1])

    mu = np.mean(x, 0)
    mu[0] = 0
    scale = np.max(x, 0) - np.min(x, 0)
    scale[0] = 1
    for i in range(0, m):
        x[i, :] = (x[i, :] - mu) / scale
    test_size = int(m/5)
    x_test = x[0:test_size, :]
    y_test = y[0:test_size, :]
    x = x[test_size:, :]
    y = y[test_size:, :]
    m -= test_size
    return m, n, x, y, x_test, y_test


def main_loop(input_filepath):
    m, n, x, y, x_test, y_test = get_data(input_filepath)
    theta = np.random.random_sample()


if __name__ == '__main__':
    params_path = input("input params file path")
    params_file = open(params_path)
    params_lines = params_file.readlines()
    data_set_num = len(params_lines)
    counter = 0
    for line in params_lines:
        main_loop(line)
        counter += 1
        print("%d of %d dataset(s) finished!", counter, data_set_num)