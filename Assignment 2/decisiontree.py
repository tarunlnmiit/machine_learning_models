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
import pandas as pd
from math import log


def main():
    # eps = np.finfo(float).eps # change it to avoid plagiarism
    args = parser.parse_args()
    file, output = args.data, args.output

    dataset = pd.read_csv(file, header=None)
    dataset.columns = ['att' + str(i) for i in range(len(dataset.iloc[0]))]

    classAttribute = dataset.columns.values[-1]
    treeClassifier(dataset, classAttribute)


def treeClassifier(dataset, classAttribute):
    classLabels = dataset[classAttribute].unique()
    if classLabels <= 1:
        return dataset[classAttribute].unique()[0]


    rootNode = findCorrectNode(dataset)
    nodeValues = dataset[rootNode].unique()

    for value in nodeValues:
        subDataset = calculateSubDataset(dataset, rootNode, value)
        nodeEntropy = calculateTotalEntropy(subDataset)

        for attribute in subDataset.drop(rootNode, axis=1).columns.values[:-1]:
            calculateAttributeEntropy(subDataset.drop(rootNode, axis=1),
                                      attribute)



def calculateSubDataset(dataset, node, value):
    subDataset = dataset[dataset[node] == value]
    return subDataset


def findCorrectNode(dataset):
    informationGainValues = []
    totalEntropy = calculateTotalEntropy(dataset)
    for attribute in dataset.columns.values[:-1]:
        attributeEntropy = calculateAttributeEntropy(dataset, attribute)
        informationGainValues.append(totalEntropy - attributeEntropy)

    maxInfoGainIndex = informationGainValues.index(max(informationGainValues))
    return dataset.columns.values[maxInfoGainIndex]


def calculateTotalEntropy(dataset):
    # calculate total entropy
    classAttribute = dataset.columns.values[-1]
    entropy = 0
    classLabels = dataset[classAttribute].unique()
    for value in classLabels:
        fraction = dataset[classAttribute].value_counts()[value] / len(
            dataset.iloc[:, -1])
        if fraction != 0 and len(classLabels) > 1:
            entropy += -fraction * log(fraction, len(classLabels))
    return entropy


def calculateAttributeEntropy(dataset, attribute):
    # calculate attribute entropy
    classAttribute = dataset.columns.values[-1]
    classLabels = dataset[classAttribute].unique()
    features = dataset[attribute].unique()
    attributeEntropy = 0
    for feature in features:
        featureEntropy = 0
        for label in classLabels:
            numerator = len(dataset[attribute][dataset[attribute] == feature][
                                dataset.iloc[:, -1] == label])
            denominator = len(dataset[attribute][dataset[attribute] == feature])
            fraction = numerator / denominator
            if fraction != 0 and len(classLabels) > 1:
                featureEntropy += -fraction * log(fraction, len(classLabels))
        attributeFraction = denominator / len(dataset)
        attributeEntropy += -attributeFraction * featureEntropy

    return abs(attributeEntropy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Data File")
    parser.add_argument("-o", "--output", help="Output")
    main()