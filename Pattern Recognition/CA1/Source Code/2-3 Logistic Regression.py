import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import math


# log normalization
def logProcessing(inputs, m, n):
    inputs_log = np.zeros([m, n])
    inputs += 0.1
    for i in range(m):
        for j in range(n):
            inputs_log[i][j] = math.log(inputs[i][j])
    return inputs_log


def dataProcessing(dataFile):
    data = scio.loadmat(dataFile)
    x_train = data['Xtrain']
    y_train = data['ytrain']
    y_train = y_train.reshape(len(y_train),)
    y_train = y_train.tolist()
    x_test = data['Xtest']
    y_test = data['ytest']
    y_test = y_test.reshape(len(y_test),)
    y_test = y_test.tolist()
    m, n = x_train.shape
    x_train_log = logProcessing(x_train, m, n)
    m, n = x_test.shape
    x_test_log = logProcessing(x_test, m, n)
    return x_train_log, y_train, x_test_log, y_test


class Logistic_Regression():
    '''
    Define a Logistic Regression Model
    Include training & test process
    Return: trained weight, accuracy of both train and test process
    '''
    def __init__(self, lamda, x_train_log, y_train):
        self.lamda = lamda
        self.x_train = x_train_log
        self.y_train = y_train

    def train(self):
        weight = np.zeros([58,1])
        ones = np.ones([3065,1])
        x_train_reg = np.c_[ones, self.x_train]
        x_t = x_train_reg.T
        EPOCH = 10
        for i in range(EPOCH):
            # Obtain u
            denominator = 1 + np.exp(-np.dot(x_train_reg, weight))
            u = 1 / denominator
            u = np.array(u)
            u = u.reshape(-1,1)
            y = np.array(self.y_train)
            y = y.reshape(-1,1)
            g = np.dot(x_t, (u-y))
            g[1:] = g[1:] + self.lamda * weight[1:]
            # Obtain S
            S = np.eye(3065)
            for m in range(3065):
                S[m][m] = u[m, 0] * (1 - u[m, 0])
            # Obtain H
            H1 = np.dot(x_t, S)
            H = np.dot(H1, x_train_reg)
            I = np.eye(57, 57)
            H[1:,1:] = H[1:,1:] + self.lamda * I
            H_inv = np.linalg.inv(H)
            weight = weight - np.dot(H_inv, g)
        return weight

    def accuracy(self, x_test_log, y_test, weight):
        # Calculate train_acc
        acc_num = 0
        x_t = self.x_train.T
        weight_t = weight.T
        for i in range(3065):
            pred = 0
            xi = x_t[:, i][:, np.newaxis]
            Denominator = 1 + np.exp(-np.dot(weight_t[:,1:], xi) - weight_t[0][0])
            pred_1 = 1/Denominator
            if pred_1[0][0] > 0.5:
                pred = 1
            if pred == self.y_train[i]:
                acc_num += 1
        train_acc = acc_num / 3065
        # Calculate test_acc
        x_t = x_test_log.T
        acc_num = 0
        for i in range(1536):
            pred = 0
            xi = x_t[:, i][:, np.newaxis]
            Denominator = 1 + np.exp(-np.dot(weight_t[:, 1:], xi) - weight_t[0][0])
            pred_1 = 1 / Denominator
            if pred_1[0][0] > 0.5:
                pred = 1
            if pred == y_test[i]:
                acc_num += 1
        test_acc = acc_num / 1536
        return train_acc, test_acc


def show_graph(lamda, err_train, err_test):
    plt.figure()
    plt.plot(lamda, err_train, label='train')
    plt.plot(lamda, err_test, label='test')
    plt.xlabel('lambda')
    plt.ylabel('Error Rates')
    plt.title('Logistic_Regression')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Change to your own file location
    dataFile = 'C:/Users/hp/Desktop/MSc 1 seme/Pattern Recognition/CA1/spamData.mat'
    x_train_log, y_train, x_test_log, y_test = dataProcessing(dataFile)
    lamda = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    err_train = []
    err_test = []
    for i in lamda:
        LR_clf = Logistic_Regression(i, x_train_log, y_train)
        weight = LR_clf.train()
        train_acc, test_acc = LR_clf.accuracy(x_test_log, y_test, weight)
        err_train.append(1-train_acc)
        err_test.append(1-test_acc)
    print('When lambda = 1, training error rate: %.2f' % (err_train[0] * 100), '%',
          'test error rate: %.2f' % (err_train[0] * 100), '%')
    print('When lambda = 10, training error rate: %.2f' % (err_train[9] * 100), '%',
          'test error rate: %.2f' % (err_test[9] * 100), '%')
    print('When lambda = 100, training error rate: %.2f' % (err_train[-1] * 100), '%',
          'test error rate: %.2f' % (err_test[-1] * 100), '%')

    show_graph(lamda, err_train, err_test)





