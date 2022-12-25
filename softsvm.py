import cvxopt.solvers
import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt


def softsvm(l, trainX: np.array, trainy: np.array):
    """
    :param l: the parameter lambda of the soft SVM algorithm
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: linear predictor w, a numpy array of size (d, 1)
    """
    # calc dimensions
    d = trainX.shape[1]
    m = trainy.shape[0]
    # init vectors v and u
    u = matrix([matrix(0.0, (d, 1)), matrix(1 / m, (m, 1))])
    v = matrix([matrix(1.0, (m, 1)), matrix(0.0, (m, 1))])
    # init blocks for block metrix A
    xy_np = np.array([trainy[i] * trainX[i] for i in range(m)])
    xy = matrix(xy_np)
    I_m = spmatrix(1.0, range(m), range(m))
    zero_MxD = spmatrix([], [], [], (m, d))
    # init A
    A = sparse([[xy, zero_MxD], [I_m, I_m]])
    # init blocks for block metrix H
    I_d = spmatrix(1.0, range(d), range(d))
    zero_DxM = spmatrix([], [], [], (d, m))
    zero_MxM = spmatrix([], [], [], (m, m))
    # init H
    H = sparse([[2 * l * I_d, zero_MxD], [zero_DxM, zero_MxM]])
    # solve
    sol = cvxopt.solvers.qp(H, u, -A, -v)
    # return only w from solution z vector
    return np.array(sol["x"][:d])


def simple_test():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100
    d = trainX.shape[1]

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvm algorithm
    w = softsvm(10, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert w.shape[0] == d and w.shape[1] == 1, f"The shape of the output should be ({d}, 1)"

    # get a random example from the test set, and classify it
    i = np.random.randint(0, testX.shape[0])
    predicty = np.sign(testX[i] @ w)

    # this line should print the classification of the i'th test sample (1 or -1).
    print(f"The {i}'th test sample was classified as {predicty}")


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # here you may add any code that uses the above functions to solve question 2
