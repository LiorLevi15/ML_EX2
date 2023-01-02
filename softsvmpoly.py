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
    G = np.block([[2 * l * gramMatrix, np.zeros((m, m))], [np.zeros((m, m)), np.zeros((m, m))]]) + epsilon * np.eye(
        2 * m)

    # print("VALS ARE:::::::")
    # for i in numpy.linalg.linalg.eigvals(G):
    #     if i <= 0:
    #         print(i)
    # print("VALS ARE:::::::")

    u = matrix([matrix(0.0, (m, 1)), matrix(1 / m, (m, 1))])
    v = matrix([matrix(0.0, (m, 1)), matrix(1.0, (m, 1))])
    subMetrix_yXG = np.zeros((m, m))
    for i in range(m):
        y_i = trainy[i]
        g_i = gramMatrix[i]
        for j in range(m):
            subMetrix_yXG[i, j] = y_i * g_i[j]
    A = sparse([[zero_mXm, matrix(subMetrix_yXG)], [I_m, I_m]])
    # print(u.trans())
    # print(u)
    # print(v)
    # print(A)
    # print(G)
    # print(G)
    # print(u.trans()*G*u)

    sol = cvxopt.solvers.qp(matrix(G), u, -A, -v)
    # print(np.array(sol["x"][:m]))
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


def plotTrainXPointByLabel(trainX: np.array, trainY: np.array):
    redPoints = [trainX[i] for i in range(len(trainY)) if trainY[i] == 1]
    bluePoints = [trainX[i] for i in range(len(trainY)) if trainY[i] == -1]
    xRedPoints = [x for x, y in redPoints]
    yRedPoints = [y for x, y in redPoints]
    xBluePoints = [x for x, y in bluePoints]
    yBluePoints = [y for x, y in bluePoints]
    plt.plot(xRedPoints, yRedPoints, 'r.')
    plt.plot(xBluePoints, yBluePoints, 'b.')
    plt.show()


def splitToCrossValidation(folds: int, trainX: np.array, trainY: np.array):
    num = int(len(trainY) / folds)
    indexes = random.sample(range(len(trainX)), num)
    validationX = [trainX[i] for i in indexes]
    validationY = [trainY[i] for i in indexes]
    newTrainX = [trainX[i] for i in range(len(trainX)) if i not in indexes]
    newTrainY = [trainY[i] for i in range(len(trainX)) if i not in indexes]
    return np.array(newTrainX), np.array(newTrainY), np.array(validationX), np.array(validationY)


def getPrediction(predictor: np.array, trainX: np.array, x: np.array, k=0):
    sum = 0
    if k != 0:
        for i, alpha_i in enumerate(predictor):
            sum += alpha_i * polynomialKernel(trainX[i], x, k)
    else:
        return np.sign(predictor.transpose() @ x)
    return np.sign(sum[0])


def getError(resY: np.array, expectedY: np.array):
    errors = 0.0
    for i in range(len(resY)):
        if resY[i] != expectedY[i]:
            errors += 1
    return float(errors / float(len(expectedY)))


def createTable(data):
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    if len(data[0]) == 3:
        df = pd.DataFrame(data, columns=["lambda_val", "k_val", "error"])
    else:
        df = pd.DataFrame(data, columns=["lambda_val", "error"])
    rows = [f"{i}" for i in range(len(data) - 1)]
    rows.append("best")
    ax.table(cellText=df.values, colLabels=df.columns,
             rowLabels=rows,
             loc='center')
    fig.tight_layout()
    plt.show()


def createMeshGrid(lVal: int, kVal: int, trainX: np.array, w: np.array):
    d = 100
    minX, maxX = min(trainX, key=lambda x: x[0]), max(trainX, key=lambda x: x[0])
    minY, maxY = min(trainX, key=lambda x: x[1]), max(trainX, key=lambda x: x[1])
    xArr = linspace(minX[0], maxX[0], d)
    yArr = linspace(minY[1], maxY[1], d)
    res = np.zeros((d, d))
    print("calculating results for mash grid")
    for i, x in enumerate(xArr):
        for j, y in enumerate(yArr):
            r = getPrediction(w, trainX, np.array([x, y]), kVal)
            res[i, j] = r

    print("ploting results")
    plt.imshow(res.transpose(),
               cmap='viridis',
               extent=[-1, 1, -1, 1],
               origin='lower'
               )
    plt.title(f"Polynomial Svm With Lambda = {lVal} and K = {kVal}")
    plt.show()


def testSvmPoly(lambdas: list, ks: list):
    data = np.load('ex2q4_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainY = data['Ytrain']
    testY = data['Ytest']
    m = 100

    smallTrainX, smallTrainY, validationX, validationY = splitToCrossValidation(5, trainX, trainY)
    data = []
    for l in lambdas:
        for k in ks:
            alpha = softsvmpoly(l, k, smallTrainX, smallTrainY)
            predictions = [getPrediction(alpha, smallTrainX, x, k=k) for x in validationX]
            error = getError(predictions, validationY)
            print(f"The (l, k) = ({l}, {k}) and the error is {error * 100}%")
            data.append([l, k, error])

    bestArgs = min(data, key=lambda x: x[2])
    l, k = bestArgs[0], bestArgs[1]
    print(l)
    print(k)
    bestPredictor = softsvmpoly(l, k, trainX, trainY)
    predictions = [getPrediction(bestPredictor, trainX, x, k) for x in testX]
    error = getError(predictions, testY)
    data.append([l, k, error])
    return data, bestPredictor, trainX


def testSvm(lambdas: list):
    data = np.load('ex2q4_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainY = data['Ytrain']
    testY = data['Ytest']
    m = 100

    smallTrainX, smallTrainY, validationX, validationY = splitToCrossValidation(5, trainX, trainY)
    data = []
    for l in lambdas:
        alpha = softsvm(l, smallTrainX, smallTrainY)
        predictions = [getPrediction(alpha, smallTrainX, x) for x in validationX]
        error = getError(predictions, validationY)
        print(f"The l = {l} and the error is {error * 100}%")
        data.append([l, error])

    bestArgs = min(data, key=lambda x: x[1])
    l = bestArgs[0]
    print(l)
    bestPredictor = softsvm(l, trainX, trainY)
    predictions = [getPrediction(bestPredictor, trainX, x) for x in testX]
    error = getError(predictions, testY)
    data.append([l, error])
    return data, bestPredictor, testX


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


def plot2dWithPredictor(alpha: np.array, l: int, k: int, trainX: np.array, allPoints: np.array):
    bluePoints = []
    redPoints = []
    for point in allPoints:
        if getPrediction(alpha, trainX, point, k) > 0:
            bluePoints.append(point)
        else:
            redPoints.append(point)
    xRedPoints = [x for x, y in redPoints]
    yRedPoints = [y for x, y in redPoints]
    xBluePoints = [x for x, y in bluePoints]
    yBluePoints = [y for x, y in bluePoints]
    plt.plot(xRedPoints, yRedPoints, 'r.')
    plt.plot(xBluePoints, yBluePoints, 'b.')
    plt.title("Best Predictor Results")
    plt.show()


def generateAllCases(t: int):
    pairs = []
    for i in range(t + 1):
        j = 0
        while j + i <= t:
            pairs.append([t - i - j, i, j])
            j += 1
    return pairs


def mulitnomial(k: int, arr: list):
    return math.factorial(k) / math.prod(math.factorial(x) for x in arr)


def getPchi(X: np.array, k: int, cases: list):
    pchiOfx = []
    for t in cases:
        b = math.sqrt(mulitnomial(k, t))

        for i, ti in enumerate(t):
            if i != 0:
                b *= math.pow(X[i - 1], ti)
        pchiOfx.append(b)
    return pchiOfx


def getVectorRepresentation(k: int, w: list):
    cases = generateAllCases(k)
    res = "["
    for i, t in enumerate(cases):
        b = format(math.sqrt(mulitnomial(k, t)) * w[i], '.2f')
        res += (f"{b}*x(1)^{t[1]}*x(2)^{t[2]}, ")
    return res + "]"


def calulateW(alpha: np.array, trainX: np.array, k: int):
    cases = generateAllCases(k)
    w = np.zeros(len(cases))
    for i, x in enumerate(trainX):
        w += alpha[i] * getPchi(x, k, cases)
    return w


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()
    # here you may add any code that uses the above functions to solve question 4
