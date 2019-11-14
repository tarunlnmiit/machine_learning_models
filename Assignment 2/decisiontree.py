#!/usr/bin/env python3
"""
[Monday 13-15] ML Programming Assignment 2
Tarun Gupta - 225900
Shipra Dureja - 225816

Command to run this decision tree model:
python3 decisiontree.py --data <filepath> --output <filename>

Input: csv File
Output: xml File
"""

import argparse
import pandas as pd
from collections import OrderedDict
from dicttoxml import dicttoxml
from math import log
from lxml import etree


def main():
    args = parser.parse_args()
    file, output = args.data, args.output

    dataset = pd.read_csv(file, header=None)
    dataset.columns = ['att' + str(i) for i in range(len(dataset.iloc[0]))]

    classAttribute = dataset.columns.values[-1]
    attributes = dataset.columns[:-1]
    logBase = len(dataset[classAttribute].unique())
    decisionTree = treeClassifier(dataset, classAttribute, attributes, logBase)
    totalEntropy = calculateNodeEntropy(dataset, logBase)

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


def treeClassifier(dataset, classAttribute, attributes, logBase):
    classLabels = dataset[classAttribute].unique()
    if len(classLabels) <= 1:
        return dataset[classAttribute].unique()[0]

    rootNode, entropy, attributeFeatureEntropies = findCorrectNode(dataset, logBase)

    decisionTree = OrderedDict()

    nodeValues = dataset[rootNode].unique()
    newAttributes = dataset.drop(rootNode, axis=1).columns.values[:-1]

    for value in nodeValues:
        subDataset = calculateSubDataset(dataset, rootNode, value)
        subBranch = treeClassifier(subDataset, classAttribute, newAttributes, logBase)

        decisionTree[rootNode + ' , ' + value + ' , ' +
                     str(attributeFeatureEntropies[rootNode][value])] = subBranch

    return decisionTree


def calculateSubDataset(dataset, node, value):
    subDataset = dataset[dataset[node] == value]
    return subDataset


def findCorrectNode(dataset, logBase):
    informationGainValues = []
    totalEntropy = calculateNodeEntropy(dataset, logBase)
    attributeFeatureEntropies = {}
    for attribute in dataset.columns.values[:-1]:
        attributeEntropy, featureEntropies = calculateAttributeEntropy(
            dataset, attribute, logBase)
        attributeFeatureEntropies[attribute] = featureEntropies
        informationGainValues.append(totalEntropy - attributeEntropy)
    maxInfoGainIndex = informationGainValues.index(max(informationGainValues))
    return dataset.columns.values[maxInfoGainIndex], totalEntropy, attributeFeatureEntropies


def calculateNodeEntropy(dataset, logBase):
    # calculate total entropy
    classAttribute = dataset.columns.values[-1]
    entropy = 0
    classLabels = dataset[classAttribute].unique()
    for value in classLabels:
        fraction = dataset[classAttribute].value_counts()[value] / len(
            dataset.iloc[:, -1])
        if fraction != 0 and len(classLabels) > 1:
            entropy -= fraction * log(fraction, logBase)
    return entropy


def calculateAttributeEntropy(dataset, attribute, logBase):
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
                featureEntropy -= fraction * log(fraction, logBase)

        attributeFraction = denominator / len(dataset)
        attributeEntropy -= attributeFraction * featureEntropy
        featureEntropies[feature] = featureEntropy

    return abs(attributeEntropy), featureEntropies


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Data File")
    parser.add_argument("-o", "--output", help="Output")
    main()
