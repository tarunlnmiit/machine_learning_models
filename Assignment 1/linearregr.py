#!/usr/bin/env python3
"""
[Monday 13-15] ML Programming Assignment 1
Tarun Gupta, Shipra Dureja

Command to run this batch gradient descent model:
python3 linearregr.py --data <filepath> --learningRate <learning Rate> --threshold <Threshold Value>

Input: csv File
Output: csv File
Output File naming convention: "solution_<input csv name>_learningRate_<learning Rate value>_threshold_<threshold value>.csv"
Output File Format(rounded to 4 decimal places): iteration_number,weight0,weight1,weight2,...,weightN,sum_of_squared_errors

The output file generated will be present in the same directory as this python file (that is parallel to this file)
"""
import argparse
import csv
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
    W = W.reshape(X.shape[1], 1).round(4)

    f_x = calculatePredicatedValue(X, W)
    sse_old = calculateSSE(Y, f_x)

    outputFile = 'solution_' + \
                 'learningRate_' + str(learningRate) + '_threshold_' \
                 + str(threshold) + '.csv'

    with open(outputFile, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, delimiter=',', quoting=csv.QUOTE_NONE, escapechar='')
        writer.writerow([*[0], *["{0:.4f}".format(val) for val in W.T[0]], *["{0:.4f}".format(sse_old)]])

        gradient, W = calculateGradient(W, X, Y, f_x, learningRate)

        iteration = 1
        while True:
            f_x = calculatePredicatedValue(X, W)
            sse_new = calculateSSE(Y, f_x)

            if abs(sse_new - sse_old) > threshold:
                writer.writerow([*[iteration], *["{0:.4f}".format(val) for val in W.T[0]], *["{0:.4f}".format(sse_new)]])
                gradient, W = calculateGradient(W, X, Y, f_x, learningRate)
                iteration += 1
                sse_old = sse_new
            else:
                break
        writer.writerow([*[iteration], *["{0:.4f}".format(val) for val in W.T[0]], *["{0:.4f}".format(sse_new)]])
    print("Output File Name: " + outputFile)


def calculateGradient(W, X, Y, f_x, learningRate):
    gradient = (Y - f_x) * X
    gradient = np.sum(gradient, axis=0)
    # gradient = np.array([float("{0:.4f}".format(val)) for val in gradient])
    temp = np.array(learningRate * gradient).reshape(W.shape)
    W = W + temp
    return gradient, W


def calculateSSE(Y, f_x):
    sse = np.sum(np.square(f_x - Y))

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