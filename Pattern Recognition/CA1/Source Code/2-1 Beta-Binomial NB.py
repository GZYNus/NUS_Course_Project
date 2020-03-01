import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import math



# Binarization normalization
def binarization(inputs, m, n):
    inputs_bi = np.zeros([m, n])
    for i in range(m):
        for j in range(n):
            if inputs[i][j] > 0:
                inputs_bi[i][j] = 1
            else:
                inputs_bi[i][j] = 0
    return inputs_bi


def dataProcessing(dataFile):
    data = scio.loadmat(dataFile)
    x_train = data['Xtrain']
    print(x_train.shape)
    print(x_train[1])
    y_train = data['ytrain']
    print(len(y_train))
    y_train = y_train.reshape(len(y_train),)
    y_train = y_train.tolist()
    x_test = data['Xtest']
    y_test = data['ytest']
    y_test = y_test.reshape(len(y_test),)
    y_test = y_test.tolist()
    m, n = x_train.shape
    x_train = binarization(x_train, m, n)
    m, n = x_test.shape
    x_test = binarization(x_test, m, n)

    return x_train, y_train, x_test, y_test


def getPrior(y_train):
    prior = {}
    prior[1] = np.sum(y_train)/len(y_train)
    prior[0] = 1 - prior[1]
    return prior


def getLikelihood(x_train, y_train):
    likelihood = [[]] * 57
    y_index = {}
    y0_index = []
    y1_index = []
    for i in range(len(y_train)):
        if y_train[i] == 0:
            y0_index.append(i)
        else:
            y1_index.append(i)
    y_index[0] = y0_index
    y_index[1] = y1_index

    for i in range(57):
        x0_index = []
        x1_index = []
        likelihood[i] = {}
        x = x_train[:,i]
        for j in range(len(x)):
            if x[j] == 0:
                x0_index.append(j)
            else:
                x1_index.append(j)
        likelihood[i][0] = x0_index
        likelihood[i][1] = x1_index
        
    like = np.ones([57,2,2])  # (feature, x, y)
    for i in range(57):
        for j in range(2):
            a = likelihood[i][j]
            for k in range(2):
                b = y_index[k]
                c = [k for k in a if k in b]
                like[i][j][k] = len(c)
    return likelihood, y_index, like


def getAlfa():
    alfa = []
    r = 0
    while r <= 100:
        alfa.append(r)
        r += 0.5
    return alfa


class naiveBayes:
    '''
    Define a Naive Bayes Classifier
    Includes training & testing process
    Return: accuracy of training & testing
    '''
    def __init__(self, alfa, likelihood, y_index):
        self.alfa = alfa
        self.likelihood = likelihood
        self.y_index = y_index

    def training(self, x_train, y_train, prior, like):
        accu_num = 0
        for i in range(len(x_train[:,0])):
            x = x_train[i]
            y0_pro = prior[0]
            y1_pro = prior[1]
            pred = 0
            # calculate y0_pro
            for j in range(57):
                index = int(x[j])  # 0 or 1
                probability = math.log((like[j][index][0] + self.alfa)/(1825 + 2 * self.alfa))
                y0_pro += probability
            # calculate y1_pro
            for j in range(57):
                index = int(x[j])  # 0 or 1
                probability = math.log((like[j][index][1] + self.alfa)/(1240 + 2 * self.alfa))
                y1_pro += probability
            if y0_pro < y1_pro:
                pred = 1
            if pred == y_train[i]:
                accu_num += 1
        acc = accu_num/3065
        return acc

    def testing(self, x_test, y_test, prior, like):
        accu_num = 0
        for i in range(len(x_test[:,0])):
            x = x_test[i]
            y0_pro = prior[0]
            y1_pro = prior[1]
            pred = 0
            # calculate y0_pro
            for j in range(57):
                index = int(x[j])  # 0 or 1
                probability = math.log((like[j][index][0] + self.alfa)/(1825 + 2 * self.alfa))
                y0_pro += probability

            # calculate y0_pro
            for j in range(57):
                index = int(x[j])  # 0 or 1
                probability = math.log((like[j][index][1] + self.alfa) / (1240 + 2 * self.alfa))
                y1_pro += probability
            if y0_pro < y1_pro:
                pred = 1
            if pred == y_test[i]:
                accu_num += 1
        acc = accu_num/1536
        return acc


def show_graph(alfa, train_err, test_err):
    plt.figure()
    plt.plot(alfa, train_err, label='train')
    plt.plot(alfa, test_err, label='test')
    plt.xlabel('alpha')
    plt.ylabel('Error Rates')
    plt.title('Naive Bayes Classifier')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Change to your own file location
    dataFile = 'C:/Users/hp/Desktop/MSc 1 seme/Pattern Recognition/CA1/spamData.mat'
    x_train, y_train, x_test, y_test = dataProcessing(dataFile)
    prior = getPrior(y_train)
    likelihood, y_index, like = getLikelihood(x_train, y_train)
    alfa = getAlfa()
    train_err = []
    test_err = []
    for i in alfa:
        print('alpha=',i)
        NB_clf = naiveBayes(i, likelihood, y_index)
        acc = NB_clf.training(x_train, y_train, prior, like)
        train_err.append(1-acc)
        acc = NB_clf.testing(x_test, y_test, prior, like)
        test_err.append(1-acc)
    print('When alpha = 1, training error rate: %.2f' % (train_err[2] * 100), '%',
          'test error rate: %.2f' % (test_err[2] * 100), '%')
    print('When alpha = 10, training error rate: %.2f' % (train_err[20] * 100), '%',
          'test error rate: %.2f' % (test_err[20] * 100), '%')
    print('When alpha = 100, training error rate: %.2f' % (train_err[-1] * 100), '%',
          'test error rate: %.2f' % (test_err[-1] * 100), '%')

    show_graph(alfa, train_err, test_err)





