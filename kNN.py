import csv
import random
import math
import operator
import  t3utility as t3
import quadtratic as quad
import numpy as np

def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(7):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def k_fold_cross_validation(X, K):
	for k in range(K):
		training = [x for i, x in enumerate(X) if i % K != k]
		validation = [x for i, x in enumerate(X) if i % K == k]
		yield training, validation

def getAccuracy(class_points, predictions):
    acc = 0
    for training_x1, validation_x1 in k_fold_cross_validation(class_points,5):
        correct = 0
        for x in range(0,len(validation_x1)):
            if validation_x1[x][-1] == predictions[x]:
                correct += 1
        acc = acc+ (correct / float(len(validation_x1)))* 100.0
    return (acc/5)

def main_cal(filename):
    # prepare data
    trainingSet = []
    testSet = []
    split = 0.67
    filepath1 = t3.currentFilePath(filename)
    dataset = np.genfromtxt(filepath1, dtype=float, delimiter=',')
    loadDataset(filename, split, trainingSet, testSet)
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))
    # generate predictions
    predictions = []
    k = 5
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))

    accuracy = getAccuracy(dataset, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')


