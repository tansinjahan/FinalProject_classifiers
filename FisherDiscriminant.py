import numpy as np
import matplotlib.pyplot as plt

def Sw(covar1, covar2):
    return covar1 + covar2

def W(Sw, M1, M2):
    Mdiff = M1 - M2
    W = np.dot(np.linalg.inv(Sw), Mdiff)
    return W

def fisherMean(W, M):
    fisherMean = np.dot(np.transpose(W), M)
    return fisherMean

def fisherVariance (W, covar):
    var = np.dot(np.dot(np.transpose(W), covar), W)
    return var

def fisherPoint(W, classPoints):
    row,col = classPoints.shape
    temp = []
    for i in range(0,row):
        trans_class = np.transpose(classPoints[i][:])
        fisherPoint = np.dot(np.transpose(W), trans_class)
        temp.append([fisherPoint,0])
    return np.asarray(temp)

def fisherThreshold(m1,m2):
    return (m1 + m2)/2

def fisherPlot(classPoints1,classPoints2,s):
    plt.scatter(classPoints1[0:,[0]],classPoints1[0:,[1]], c ="red")
    plt.scatter(classPoints2[0:, [0]], classPoints2[0:, [1]], c = "blue")
    plt.title("Fishers Discriminant " + str(s) + " diagonalization")
    plt.show()
