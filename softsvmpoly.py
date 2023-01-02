import math
import random
import pandas as pd
import cvxopt
import numpy as np
import numpy.linalg.linalg
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt
from numpy import linspace
from softsvm import softsvm

"""
    :param l: the parameter lambda of the soft SVM algorithm
    :param sigma: the bandwidth parameter sigma of the RBF kernel.
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: numpy array of size (m, 1) which describes the coefficients found by the algorithm
   """


def softsvmpoly(l: float, k: int, trainX: np.array, trainy: np.array):
    epsilon = 0.0001
    m = trainX.shape[0]
    d = trainX.shape[1]
    gramMatrix = getGramMatrix(trainX, k)
    I_m = spmatrix(1.0, range(m), range(m))
    zero_mXm = spmatrix(0.0, range(m), range(m))
    G = np.block([[2 * l * gramMatrix, np.zeros((m, m))], [np.zeros((m, m)), np.zeros((m, m))]]) + epsilon * np.eye(2 * m)
    u = matrix([matrix(0.0, (m, 1)), matrix(1 / m, (m, 1))])
    v = matrix([matrix(0.0, (m, 1)), matrix(1.0, (m, 1))])
    subMetrix_yXG = np.zeros((m, m))
    for i in range(m):
        y_i = trainy[i]
        g_i = gramMatrix[i]
        for j in range(m):
            subMetrix_yXG[i, j] = y_i * g_i[j]
    A = sparse([[zero_mXm, matrix(subMetrix_yXG)], [I_m, I_m]])

    sol = cvxopt.solvers.qp(matrix(G), u, -A, -v)
    return np.array(sol["x"][:m])


def polynomialKernel(x1: np.array, x2: np.array, k: int):
    return pow((1 + x1 @ x2), k)


def getGramMatrix(trainX: np.array, k: int):
    m = trainX.shape[0]
    gramMatrix = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            gramMatrix[i, j] = polynomialKernel(trainX[i], trainX[j], k)
    return gramMatrix


def simple_test():
    # load question 2 data
    data = np.load('ex2q4_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # plotTrainXPointByLabel(_trainX, _trainy)
    # # run the softsvmpoly algorithm
    w = softsvmpoly(10, 5, _trainX, _trainy)
    #
    # # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvmbf should be a numpy array"
    assert w.shape[0] == m and w.shape[1] == 1, f"The shape of the output should be ({m}, 1)"


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()
    # here you may add any code that uses the above functions to solve question 4
