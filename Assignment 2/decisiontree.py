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
import pandas as pd
from collections import OrderedDict
from dicttoxml import dicttoxml
from math import log
from pprint import pprint
from lxml import etree
from xml.etree import ElementTree


def main():
    args = parser.parse_args()
    file, output = args.data, args.output

    dataset = pd.read_csv(file, header=None)
    dataset.columns = ['att' + str(i) for i in range(len(dataset.iloc[0]))]

    classAttribute = dataset.columns.values[-1]
    attributes = dataset.columns[:-1]
    decisionTree = treeClassifier(dataset, classAttribute, attributes)
    totalEntropy = calculateNodeEntropy(dataset)
    # pprint(decisionTree)

    xml = dicttoxml(decisionTree, custom_root='tree', attr_type=False)
    with open(output, 'wb') as file:
        file.write(xml)

    tree = etree.parse(output)
    root = tree.getroot()
    root.set('entropy', str(totalEntropy))

    for node in root.iter('key'):
        node.tag = 'node'
        attributeValue = node.attrib['name'].split(' , ')
        feature = attributeValue[0]
        value = attributeValue[1]
        entropy = attributeValue[2]
        node.set('entropy', entropy)
        node.set('feature', feature)
        node.set('value', value)
        node.attrib.pop('name', None)

    tree = etree.ElementTree(root)
    tree.write(output, xml_declaration=False)


def treeClassifier(dataset, classAttribute, attributes):
    classLabels = dataset[classAttribute].unique()
    if len(classLabels) <= 1:
        return dataset[classAttribute].unique()[0]

    rootNode, entropy, attributeFeatureEntropies = findCorrectNode(dataset)
    print(attributeFeatureEntropies, entropy)

    decisionTree = OrderedDict()
    # decisionTree[rootNode] = OrderedDict()

    nodeValues = dataset[rootNode].unique()
    newAttributes = dataset.drop(rootNode, axis=1).columns.values[:-1]

    for value in nodeValues:
        subDataset = calculateSubDataset(dataset, rootNode, value)
        subBranch = treeClassifier(subDataset, classAttribute, newAttributes)

        decisionTree[rootNode + ' , ' + value + ' , ' +
                     str(attributeFeatureEntropies[rootNode][value])] = subBranch

    return decisionTree


def calculateSubDataset(dataset, node, value):
    subDataset = dataset[dataset[node] == value]
    return subDataset


def findCorrectNode(dataset):
    informationGainValues = []
    totalEntropy = calculateNodeEntropy(dataset)
    attributeFeatureEntropies = {}
    for attribute in dataset.columns.values[:-1]:
        attributeEntropy, featureEntropies = calculateAttributeEntropy(
            dataset, attribute)
        attributeFeatureEntropies[attribute] = featureEntropies
        informationGainValues.append(totalEntropy - attributeEntropy)
    # print(informationGainValues)
    maxInfoGainIndex = informationGainValues.index(max(informationGainValues))
    return dataset.columns.values[maxInfoGainIndex], totalEntropy, attributeFeatureEntropies


def calculateNodeEntropy(dataset):
    # calculate total entropy
    classAttribute = dataset.columns.values[-1]
    entropy = 0
    classLabels = dataset[classAttribute].unique()
    for value in classLabels:
        fraction = dataset[classAttribute].value_counts()[value] / len(
            dataset.iloc[:, -1])
        if fraction != 0 and len(classLabels) > 1:
            entropy += -fraction * log(fraction, len(classLabels))
    # print('node', entropy)
    return entropy


def calculateAttributeEntropy(dataset, attribute):
    # calculate attribute entropy
    classAttribute = dataset.columns.values[-1]
    classLabels = dataset[classAttribute].unique()
    features = dataset[attribute].unique()
    attributeEntropy = 0
    featureEntropies = {}
    for feature in features:
        featureEntropy = 0
        for label in classLabels:
            numerator = len(dataset[attribute][dataset[attribute] == feature][
                dataset.iloc[:, -1] == label])
            denominator = len(dataset[attribute]
                              [dataset[attribute] == feature])
            fraction = numerator / denominator
            if fraction != 0 and len(classLabels) > 1:
                featureEntropy += -fraction * log(fraction, len(classLabels))
        attributeFraction = denominator / len(dataset)
        attributeEntropy += -attributeFraction * featureEntropy
        featureEntropies[feature] = featureEntropy
        # print('f', featureEntropy, 'a', attributeEntropy)

    return abs(attributeEntropy), featureEntropies


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Data File")
    parser.add_argument("-o", "--output", help="Output")
    main()
