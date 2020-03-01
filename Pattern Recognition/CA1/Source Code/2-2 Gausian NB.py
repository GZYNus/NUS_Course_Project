import scipy.io as scio
import numpy as np
import math
# import xgboost as xgb



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


def get_mean_var(x_train_log, y_train):
    # Calculate mean & variance of y=1 & y= 0 separately
    mean0 = []
    mean1 = []
    var0 = []
    var1 = []
    index0 = []
    index1 = []
    x_train_y0 = np.zeros([1,57])
    x_train_y1 = np.zeros([1,57])
    for i in range(len(y_train)):
        if y_train[i] == 0:
            index0.append(i)
        else:
            index1.append(i)
    for i in index0:
        m = x_train_log[i][np.newaxis,:]
        x_train_y0 = np.r_[x_train_y0, m]
    for i in index1:
        m = x_train_log[i][np.newaxis, :]
        x_train_y1 = np.r_[x_train_y1, m]
    x_train_y0 = np.delete(x_train_y0, 0, axis=0)
    x_train_y1 = np.delete(x_train_y1, 0, axis=0)
    for i in range(57):
        x = x_train_y0[:,i]
        mean0.append(np.mean(x))
        var0.append(np.var(x))
        x = x_train_y1[:,i]
        mean1.append(np.mean(x))
        var1.append(np.var(x))
    return mean0, mean1, var0, var1


def getPrior(y_train):
    prior = {}
    prior[1] = np.sum(y_train)/len(y_train)
    prior[0] = 1 - prior[1]
    return prior


class Gausian_NB:
    '''
    Define a Naive Bayes Classifier
    Include training & test process
    The features are all subject to Normal Distribution
    Return: accuracy of training and testing phases
    '''
    def __init__(self,  mean0, mean1, var0, var1, prior):
        self.mean0 = mean0
        self.mean1 = mean1
        self.var0 = var0
        self.var1 = var1
        self.prior = prior

    def training(self, x_train_log, y_train):
        acc_num = 0
        bias = 10e-30
        for i in range(len(x_train_log[:,0])):
            x = x_train_log[i]
            y0_pro = prior[0]
            y1_pro = prior[1]
            pred = 0
            # calculate y0_pro
            for j in range(57):
                index = x[j]
                y_pro = np.exp(-(index - self.mean0[j]) ** 2 / (2 * self.var0[j])) / (
                    math.sqrt(2 * math.pi * self.var0[j]))
                probability = math.log(y_pro+bias)
                y0_pro += probability
            # calculate y1_pro
            for j in range(57):
                index = x[j]
                y_pro = np.exp(-(index - self.mean1[j]) ** 2 / (2 * self.var1[j])) / (
                    math.sqrt(2 * math.pi * self.var1[j]))
                probability = math.log(y_pro+bias)
                y1_pro += probability
            if y0_pro < y1_pro:
                pred = 1
            if pred == y_train[i]:
                acc_num += 1
        acc = acc_num / len(x_train_log[:,0])
        return acc

    def testing(self, x_test_log, y_test):
        acc_num = 0
        bias = 10e-30
        for i in range(len(x_test_log[:, 0])):
            x = x_test_log[i]
            y0_pro = prior[0]
            y1_pro = prior[1]
            pred = 0
            # calculate y0_pro
            for j in range(57):
                index = x[j]
                y_pro = np.exp(-(index - self.mean0[j]) ** 2 / (2 * self.var0[j])) / (
                    math.sqrt(2 * math.pi * self.var0[j]))
                probability = math.log(y_pro+bias)
                y0_pro += probability
            # calculate y1_pro
            for j in range(57):
                index = x[j]
                y_pro = np.exp(-(index - self.mean1[j]) ** 2 / (2 * self.var1[j])) / (
                    math.sqrt(2 * math.pi * self.var1[j]))
                probability = math.log(y_pro+bias)
                y1_pro += probability
            if y0_pro < y1_pro:
                pred = 1
            if pred == y_test[i]:
                acc_num += 1
        acc = acc_num / len(x_test_log[:, 0])
        return acc


if __name__ == '__main__':
    # Change to your own file location
    dataFile = 'C:/Users/hp/Desktop/MSc 1 seme/Pattern Recognition/CA1/spamData.mat'
    x_train_log, y_train, x_test_log, y_test = dataProcessing(dataFile)
    mean0, mean1, var0, var1 = get_mean_var(x_train_log, y_train)
    prior = getPrior(y_train)
    NB_clf = Gausian_NB( mean0, mean1, var0, var1, prior)
    acc_train = NB_clf.training(x_train_log, y_train)
    acc_test = NB_clf.testing(x_test_log, y_test)
    train_err = 1 - acc_train
    test_err = 1 - acc_test
    print('train_error:',train_err)
    print('test_error:',test_err)




