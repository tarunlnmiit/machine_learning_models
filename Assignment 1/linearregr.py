import argparse
import csv
from decimal import Decimal
import numpy as np


def main():
    # Shiprs's github profile -- ShipraDureja
    args = parser.parse_args()
    file, learningRate, threshold = args.data, float(
        args.learningRate), float(args.threshold)
    with open(file) as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        X = []
        Y = []
        for row in reader:
            X.append([1] + row[:-1])
            Y.append(row[-1])

    n = len(X)
    X = np.array(X).astype(float)
    Y = np.array(Y).astype(float)
    W = np.zeros(len(X[0])).astype(float)
    # W[0] = 1

    """ Feature Scaling
    mean = np.ones(X.shape[1] - 1)
    std = np.ones(X.shape[1] - 1)

    for i in range(1, X.shape[1]):
        mean[i-1] = np.mean(X.transpose()[i])
        std[i-1] = np.std(X.transpose()[i])
        for j in range(X.shape[0]):
            X[j][i] = (X[j][i] - mean[i-1]) / std[i-1]
    """

    f_x = np.ones(X.shape[0])
    sse = 'inf'
    iteration = 1
    while sse == 'inf' or threshold < sse:
        for i in range(n):
            f_x[i] = float(np.matmul(W, X[i]))

        sse = np.sum(np.square(Y - f_x))
        # sse = (1 / X.shape[0]) * 0.5 * sum(np.square(Y - f_x))
        sse = float("{0:.4f}".format(sse))
        gradient = np.ones((X.shape[0], len(X[0])))
        for i in range(n):
            gradient[i] = (Y[i] - f_x[i]) * X[i]

        print("{}, {}, {}" .format(iteration, ', '.join(
            ["{0:.4f}".format(val) for val in W]), "{0:.4f}".format(sse)))

        gradient = np.sum(gradient, axis=0)
        W += learningRate * gradient
        W = np.array([float("{0:.4f}".format(val)) for val in W])

        iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Data File")
    parser.add_argument("-l", "--learningRate", help="Learning Rate")
    parser.add_argument("-t", "--threshold", help="Threshold")
    main()
