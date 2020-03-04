from svmutil import *
from svm import *
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

random.seed(100)


def dataloding():
    image_dataset = np.fromfile('image_array.bin', dtype=np.float64)
    label = np.fromfile('label.bin', dtype=int)
    image_dataset = image_dataset.reshape(3400, -1)
    return image_dataset, label


def myLoad():
    my_dataset = np.fromfile('my_array.bin', dtype=np.float64)
    my_label = np.fromfile('my_label.bin',dtype=int)
    my_dataset = my_dataset.reshape(10,-1)
    return my_dataset, my_label


# Create PCA model
def pca(XMat, k):
    average = np.mean(XMat)
    m, n = np.shape(XMat)
    avgs = np.tile(average, (m, 1))
    data_adjust = XMat - avgs
    covX = np.cov(data_adjust.T)   # cal covariance matrix
    featValue, featVec = np.linalg.eig(covX)    # cal eigenvalue & eigenvector
    index = np.argsort(-featValue)  # descending sort
    if k > n:
        print("k must be smaller than feature number")
        return
    else:
        selectVec = np.array(featVec.T[index[:k]])
        finalData = np.dot(data_adjust, selectVec.T)
    return selectVec, finalData


def pca_transform(XMat, selectVec):
    average = np.mean(XMat)
    m, n = np.shape(XMat)
    avgs = np.tile(average, (m, 1))
    data_adjust = XMat - avgs
    finalData = np.dot(data_adjust, selectVec.T)
    return finalData


def show_graph(C, acc):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    train_acc = acc[80]['train_acc']
    test_acc = acc[80]['test_acc']
    ax1.plot(C, train_acc, label='train')
    ax1.plot(C, test_acc, label='test')
    plt.xlabel('C Value')
    plt.ylabel('Accuracy Rates')
    plt.title('SVM with 80 PCs')
    ax1.legend()

    ax2 = fig.add_subplot(122)
    train_acc = acc[200]['train_acc']
    test_acc = acc[200]['test_acc']
    ax2.plot(C, train_acc, label='train')
    ax2.plot(C, test_acc, label='test')
    plt.xlabel('C Value')
    plt.ylabel('Accuracy Rates')
    plt.title('SVM with 200 PCs')
    ax2.legend()

    plt.show()


def train_and_predict(PC, C,  x_train, x_test, y_train, y_test):
    acc = {}
    for i in PC:
        acc[i] = {}
        train_acc = []
        test_acc = []
        selectVec, train_tran = pca(x_train, i)
        test_tran = pca_transform(x_test, selectVec)
        # train_tran = x_train
        # test_tran = x_test
        for j in C:
            prob = svm_problem(y_train, train_tran)
            string = '-t 0 -c ' + str(j) + ' -q'
            param = svm_parameter(string)
            model = svm_train(prob, param)
            # print('Train:')
            p_label, p_acc, p_val = svm_predict(y_train, train_tran, model)
            train_acc.append(p_acc[0])

            # print('Test:')
            p_label, p_acc, p_val = svm_predict(y_test, test_tran, model)
            test_acc.append(p_acc[0])
        acc[i]['train_acc'] = train_acc
        acc[i]['test_acc'] = test_acc
    show_graph(C, acc)


if __name__ == '__main__':
    image_dataset, label = dataloding()
    x_train, x_test, y_train, y_test = train_test_split(image_dataset, label, test_size=0.3, random_state=10)
    my_pic, my_label = myLoad()
    # y_train = np.concatenate((y_train, my_label[3:]), axis=0)
    # x_train = np.concatenate((x_train, my_pic[3:]), axis=0)
    # x_test = my_pic[:3]
    # y_test = my_label[:3]
    PC = [80, 200]
    C = [0.01, 0.1, 1]
    train_and_predict(PC, C,  x_train, x_test, y_train, y_test)









