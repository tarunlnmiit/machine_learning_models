import argparse
import csv
from decimal import Decimal
import numpy as np


def main():
    args = parser.parse_args()
    file, learningRate, threshold = args.data, float(
        args.learningRate), float(args.threshold)
    with open(file) as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        X = []
        Y = []
        for row in reader:
            X.append([1.0] + row[:-1])
            Y.append([row[-1]])

    n = len(X)
    X = np.array(X).astype(float)
    Y = np.array(Y).astype(float)
    W = np.zeros(X.shape[1]).astype(float)
    W = W.reshape(X.shape[1], 1)

    f_x = calculatePredicatedValue(X, W)
    sse_old = calculateSSE(Y, f_x)

    print("{}, {}, {}".format(0, ', '.join(
        ["{0:.4f}".format(val) for val in W.T[0]]), sse_old))

    gradient, W = calculateGradient(W, X, Y, f_x, learningRate)

    iteration = 1
    while True:
        f_x = calculatePredicatedValue(X, W)
        sse_new = calculateSSE(Y, f_x)

        if (sse_new - sse_old) == 0:
            break
        if (sse_new - sse_old) <= threshold:
            print("{}, {}, {}".format(iteration, ', '.join(
                ["{0:.4f}".format(val) for val in W.T[0]]), sse_new))
            gradient, W = calculateGradient(W, X, Y, f_x, learningRate)
            iteration += 1
            sse_old = sse_new
        else:
            break
    print("{}, {}, {}".format(iteration, ', '.join(
        ["{0:.4f}".format(val) for val in W.T[0]]), sse_new))

def calculateGradient(W, X, Y, f_x, learningRate):
    gradient = (Y - f_x) * X
    gradient = np.sum(gradient, axis=0)
    temp = np.array(learningRate * gradient).reshape(W.shape)
    W = W + temp
    return gradient, W


def calculateSSE(Y, f_x):
    sse = np.sum(np.square(f_x - Y))
    sse = float("{0:.4f}".format(sse))

    return sse


def calculatePredicatedValue(X, W):
    f_x = np.dot(X, W)
    return f_x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Data File")
    parser.add_argument("-l", "--learningRate", help="Learning Rate")
    parser.add_argument("-t", "--threshold", help="Threshold")
    main()