import numpy as np
import matplotlib.pyplot as plt
import softsvm

# load question 2 data
data = np.load('EX2q2_mnist.npz')
trainX = data['Xtrain']
testX = data['Xtest']
trainY = data['Ytrain']
testY = data['Ytest']


def getTrainSample(size: int):
    # Get a random size training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:size]]
    _trainY = trainY[indices[:size]]
    return _trainX, _trainY


def predictAndCalcError(l, _trainX: np.array, _trainY: np.array, _testX: np.array, _testY: np.array):
    w = softsvm.softsvm(l, _trainX, _trainY)
    preds = []
    for x in _testX:
        if np.dot(w.reshape(w.shape[0]), x) > 0:
            preds.append(1)
        else:
            preds.append(-1)
    return np.mean(preds != _testY)


def runTestCase(ls, sampleSize: int, title: str, reps=10, draw_lines=True, draw_minmax_bars=True):
    train_errors = [np.zeros(reps, float) for _ in ls]
    test_errors = [np.zeros(reps, float) for _ in ls]
    for idx, l in enumerate(ls):
        for i in range(reps):
            _trainX, _trainY = getTrainSample(sampleSize)
            train_errors[idx][i] = predictAndCalcError(10 ** l, _trainX, _trainY, _trainX, _trainY)
            test_errors[idx][i] = predictAndCalcError(10 ** l, _trainX, _trainY, testX, testY)

    train_mean_errors = [error.mean() for error in train_errors]
    train_min_errors = [error.min() for error in train_errors]
    train_max_errors = [error.max() for error in train_errors]
    test_mean_errors = [error.mean() for error in test_errors]
    test_min_errors = [error.min() for error in test_errors]
    test_max_errors = [error.max() for error in test_errors]

    w = 0.1
    if draw_lines:
        plt.plot(ls, train_mean_errors, label='average train error')
        plt.plot(ls, test_mean_errors, label='average test error')
    else:
        plt.plot(ls, train_mean_errors, label='average train error', marker="o", linestyle="None")
        plt.plot(ls, test_mean_errors, label='average test error', marker="o", linestyle="None")
    if draw_minmax_bars:
        plt.bar([l-w for l in ls], train_max_errors, width=w,  label='max train error')
        plt.bar(ls, train_min_errors, width=w, label='min train error')
        plt.bar([l+w for l in ls], test_max_errors, width=w, label='max test error')
        plt.bar([l+2*w for l in ls], test_min_errors, width=w, label='min test error')

    plt.title(title)
    plt.xlabel("lambdas")
    plt.ylabel("errors")
    plt.legend()
    plt.show()


def task2a():
    ls = [n for n in range(1, 11)]
    runTestCase(ls, 100, "task2a")


def task2b():
    ls = [1, 3, 5, 8]
    runTestCase(ls, 1000, "task2b", reps=1, draw_lines=False, draw_minmax_bars=False)


if __name__ == '__main__':
    # task2a()
    task2b()
