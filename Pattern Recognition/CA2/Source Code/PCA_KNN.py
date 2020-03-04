from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import random
import heapq


def dataLoading():
    image_dataset = np.fromfile('image_array.bin',dtype=np.float64)
    label = np.fromfile('label.bin', dtype= int)
    image_dataset = image_dataset.reshape(3400, -1)
    # split train & test with the ratio of 0.7
    x_train, x_test, y_train, y_test = train_test_split(image_dataset, label, test_size=0.3, random_state=10)
    return x_train, x_test, y_train, y_test


def myLoad():
    my_dataset = np.fromfile('my_array.bin', dtype=np.float64)
    my_label = np.fromfile('my_label.bin',dtype=int)
    my_dataset = my_dataset.reshape(10,-1)
    return my_dataset, my_label


class knn_classifier:
    """
    Define a knn_classifier class
    Includes classification process
    Suitable for both training and testing process
    Parameter: K, data and label
    Return: accuracy
    """
    def __init__(self, K):
        self.k = K

    def clf(self, x_data, y_label):
        m, n = x_data.shape
        acc = 0
        for i in range(m):
            x = x_data[i]
            euclidean_distance = np.sqrt(np.sum(np.square(x_data - x), axis=1))
            index = heapq.nsmallest(self.k, range(len(euclidean_distance)), euclidean_distance.take)
            yhat = y_label[index]
            yhat = yhat.tolist()
            y_pred = Counter(yhat).most_common(1)[0][0]
            if y_pred == y_label[i]:
                acc += 1
        return acc / m

    def pred_clf(self, x_train, y_train, x_test, y_test):
        m, n = x_test.shape
        acc = 0
        for i in range(m):
            x = x_test[i]
            euclidean_distance = np.sqrt(np.sum(np.square(x_train - x), axis=1))
            index = heapq.nsmallest(self.k, range(len(euclidean_distance)), euclidean_distance.take)
            yhat = y_train[index]
            yhat = yhat.tolist()
            y_pred = Counter(yhat).most_common(1)[0][0]
            if y_pred == y_test[i]:
                acc += 1
        return acc/m


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
        selectVec = np.array(featVec.T[index[:k]])  # pay attention to transpose
        finalData = np.dot(data_adjust, selectVec.T)
    return selectVec, finalData


def pca_transform(XMat, selectVec):
    average = np.mean(XMat)
    m, n = np.shape(XMat)
    avgs = np.tile(average, (m, 1))
    data_adjust = XMat - avgs
    finalData = np.dot(data_adjust, selectVec.T)
    return finalData


def show_graph(k, acc):
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    train_acc = acc[40]['train_acc']
    test_acc = acc[40]['test_acc']
    ax1.plot(k, train_acc, label='train')
    ax1.plot(k, test_acc, label='test')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy Rates')
    plt.title('PCA with 40 PCs')
    ax1.legend()

    ax2 = fig.add_subplot(132)
    train_acc = acc[80]['train_acc']
    test_acc = acc[80]['test_acc']
    ax2.plot(k, train_acc, label='train')
    ax2.plot(k, test_acc, label='test')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy Rates')
    plt.title('PCA with 80 PCs')
    ax2.legend()

    ax3 = fig.add_subplot(133)
    train_acc = acc[200]['train_acc']
    test_acc = acc[200]['test_acc']
    ax3.plot(k, train_acc, label='train')
    ax3.plot(k, test_acc, label='test')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy Rates')
    plt.title('PCA with 200 PCs')
    ax3.legend()

    plt.show()


def printAccuracy(PC, k_number, x_train, x_test, y_train, y_test):
    acc = {}
    for i in PC:
        acc[i] = {}
        train_acc_list = []
        test_acc_list = []
        print('------------------------')
        print('Number of PCs is:',i)
        selectVec, train_transData = pca(x_train, i)
        test_transData = pca_transform(x_test, selectVec)
        for j in k_number:
            print('When consider', j, 'neighbors:')
            # Calculate training accuracy
            KNN = knn_classifier(j)
            train_acc = KNN.clf(train_transData, y_train)
            train_acc_list.append(train_acc)
            # Calculate test accuracy
            test_acc = KNN.pred_clf(train_transData, y_train, test_transData, y_test)
            test_acc_list.append(test_acc)
            print('Testing accuracy:', test_acc)
        acc[i]['train_acc'] = train_acc_list
        acc[i]['test_acc'] = test_acc_list

    show_graph(k_number, acc)


if __name__ == '__main__':
    random.seed(1000)
    x_train, x_test, y_train, y_test = dataLoading()
    my_pic, my_label = myLoad()
    PC = [40, 80, 200]
    # Suppose the number of nearest neighbors
    k_number = [1,3]
    printAccuracy(PC, k_number, x_train, x_test, y_train, y_test)














