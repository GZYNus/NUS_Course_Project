import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import math
import heapq


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
    x_test = data['Xtest']
    y_test = data['ytest']
    m, n = x_train.shape
    x_train_log = logProcessing(x_train, m, n)
    m, n = x_test.shape
    x_test_log = logProcessing(x_test, m, n)
    return x_train_log, y_train, x_test_log, y_test


class knn_classifier:
    '''
    Define a knn_classifier class
    Including training & testing precess
    '''
    def __init__(self, K, x_train_log, y_train):
        self.k = K
        self.x_train = x_train_log
        self.y_train = y_train

    # Explanation:
    # Calculate Euclidean_distance between the particular point & all points

    def training(self):
        m, n = self.x_train.shape
        train_label = []
        for i in range(m):
            x = self.x_train[i]
            euclidean_distance = np.sqrt(np.sum(np.square(self.x_train - x), axis=1))
            index = heapq.nsmallest(self.k, range(len(euclidean_distance)),euclidean_distance.take)
            y_sum = np.sum(self.y_train[index])
            if (y_sum/self.k) > 0.5:
                train_label.append(1)
            else:
                train_label.append(0)
        y_label = self.y_train.reshape(m,)
        y_label = y_label.tolist()
        acc = 0
        for i in range(len(y_label)):
            if train_label[i] == y_label[i]:
                acc += 1
        return acc/len(y_label)

    def testing(self, x_test, y_test):
        m, n = x_test.shape
        test_label = []
        for i in range(m):
            x = x_test[i]
            euclidean_distance = np.sqrt(np.sum(np.square(self.x_train - x), axis=1))
            index = heapq.nsmallest(self.k, range(len(euclidean_distance)), euclidean_distance.take)
            y_sum = np.sum(self.y_train[index])
            if (y_sum/self.k) > 0.5:
                test_label.append(1)
            else:
                test_label.append(0)
        y_label = y_test.reshape(m,)
        y_label = y_label.tolist()
        acc = 0
        for i in range(len(y_label)):
            if test_label[i] == y_label[i]:
                acc += 1
        return acc/len(y_label)


def show_graph(K, train_err, test_err):
    plt.figure()
    plt.plot(K, train_err, label='train')
    plt.plot(K, test_err, label='test')
    plt.xlabel('K Value')
    plt.ylabel('Error Rates')
    plt.title('KNN Classifier')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Change to your own file location
    dataFile = 'C:/Users/hp/Desktop/MSc 1 seme/Pattern Recognition/CA1/spamData.mat'
    x_train_log, y_train, x_test_log, y_test = dataProcessing(dataFile)
    K = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    train_err = []
    test_err = []
    for k in K:
        print('Now calculate K:',k)
        KNN = knn_classifier(k, x_train_log, y_train)
        acc = KNN.training()
        train_err.append(1-acc)
        acc = KNN.testing(x_test_log, y_test)
        test_err.append(1-acc)
    print('When K = 1, training error rate: %.2f' % (train_err[0]*100), '%',
          'test error rate: %.2f' % (test_err[0]*100),'%')
    print('When K = 10, training error rate: %.2f' % (train_err[9] * 100), '%',
          'test error rate: %.2f' % (test_err[9] * 100), '%')
    print('When K = 100, training error rate: %.2f' % (train_err[-1] * 100), '%',
          'test error rate: %.2f' % (test_err[-1] * 100), '%')

    show_graph(K, train_err, test_err)




