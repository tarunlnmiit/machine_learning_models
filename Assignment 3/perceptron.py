#!/usr/bin/env python3
"""
[Monday 13-15] ML Programming Assignment 3
Tarun Gupta, Shipra Dureja

Command to run this batch gradient descent model:
python3 perceptron.py --data <filepath> --output <output file>

Input: tsv File
Output: tsv File
"""
import argparse
import csv
import numpy as np


def main():
    args = parser.parse_args()
    file, outputFile = args.data, args.output
    learningRate = 1
    with open(file) as tsvFile:
        reader = csv.reader(tsvFile, delimiter='\t')
        X = []
        Y = []
        for row in reader:
            if row[-1] == '':
                X.append([1.0] + row[1:-1])
            else:
                X.append([1.0] + row[1:])
            if row[0] == 'A':
                Y.append([1])
            else:
                Y.append([0])

    n = len(X)
    X = np.array(X).astype(float)
    Y = np.array(Y).astype(float)
    W = np.zeros(X.shape[1]).astype(float)
    W = W.reshape(X.shape[1], 1).astype(float)

    normalError = calculateNormalBatchLearning(X, Y, W, learningRate)
    annealError = calculateAnnealBatchLearning(X, Y, W, learningRate)

    with open(outputFile, 'w') as tsvFile:
        writer = csv.writer(tsvFile, delimiter='\t')
        writer.writerow(normalError)
        writer.writerow(annealError)


def calculateNormalBatchLearning(X, Y, W, learningRate):
    e = []
    for i in range(101):
        f_x = calculatePredicatedValue(X, W)
        errorCount = calculateError(Y, f_x)
        e.append(errorCount)
        gradient, W = calculateGradient(W, X, Y, f_x, learningRate)
    return e


def calculateAnnealBatchLearning(X, Y, W, learningRate):
    e = []
    for i in range(101):
        f_x = calculatePredicatedValue(X, W)
        errorCount = calculateError(Y, f_x)
        e.append(errorCount)
        learningRate = 1 / (i + 1)
        gradient, W = calculateGradient(W, X, Y, f_x, learningRate)
    return e


def calculateGradient(W, X, Y, f_x, learningRate):
    gradient = (Y - f_x) * X
    gradient = np.sum(gradient, axis=0)
    # gradient = np.array([float("{0:.4f}".format(val)) for val in gradient])
    temp = np.array(learningRate * gradient).reshape(W.shape)
    W = W + temp
    return gradient, W.astype(float)


def calculateError(Y, f_x):
    errorCount = 0
    for i in range(len(f_x)):
        if Y[i][0] != f_x[i][0]:
            errorCount += 1

    return errorCount


def calculatePredicatedValue(X, W):
    f_x = np.dot(X, W)
    for i in range(len(f_x)):
        if f_x[i][0] > 0:
            f_x[i][0] = 1
        else:
            f_x[i][0] = 0
    return f_x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Data File")
    parser.add_argument("-o", "--output", help="output")
    main()