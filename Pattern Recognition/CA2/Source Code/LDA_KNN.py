import random
import matplotlib.pyplot as plt
import heapq
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter


# load images and labels and generate training set, including randomly selected 500 images and labels
def dataLoading():
    image_dataset = np.fromfile('image_array.bin',dtype=np.float64)
    label = np.fromfile('label.bin', dtype= int)
    image_dataset = image_dataset.reshape(3400, -1)
    # split train & test with the ratio of 0.7
    x_train, x_test, y_train, y_test = train_test_split(image_dataset, label, test_size=0.3, random_state=100)
    x_train = x_train * 255
    x_test = x_test * 255
    # print(x_train.shape)                                              # shape:(2380,1024)
    return x_train, x_test, y_train, y_test


def myLoad():
    my_dataset = np.fromfile('my_array.bin', dtype=np.float64)
    my_label = np.fromfile('my_label.bin',dtype=int)
    my_dataset = my_dataset.reshape(10,-1)
    return my_dataset, my_label


def cal_label(y_train):
    classes = list(set(y_train))
    num_class = len(classes)                                            # int: 20
    num_each_class = dict(Counter(y_train))
    num_each_class = sorted(num_each_class.items(), key=lambda x:x[0])   # format: [(1, 26), (2, 27), (3, 34)]
    return num_class, num_each_class


def cal_mu_sum(x_mat):
    return np.mean(x_mat,0)


def compute_ss(x_train, y_train):
    ss = {}
    for i in range(NUM_Labels):
        i += 1
        ss[i] = []
    for i in range(sum_pics):
        a = y_train[i]
        ss[a].append(x_train[i])
    for i in range(NUM_Labels):
        i += 1
        ss_sub = np.array(ss[i])
        ss[i] = ss_sub
    return ss


def cal_mu_i(ss, num_class):
    mu_i = []
    for i in range(num_class):
        i += 1
        a = np.mean(ss[i],0)
        mu_i.append(a)

    return mu_i


def cal_sb(num_each_class, mu_i, mu_sum):
    sb_i = {}
    prob = []
    for i in range(NUM_Labels):
        prob.append(num_each_class[i][1]/sum_pics)

    for i in range(NUM_Labels):
        j = i + 1
        aa = np.array(mu_i[i]-mu_sum)
        aa = aa.reshape(1,-1)
        sb_i[j] = prob[i] * np.dot(aa.T,aa)

    Sb = 0
    for i in range(NUM_Labels):
        i += 1
        Sb += sb_i[i]

    # print(Sb.shape)                                                   # shape: (1024,1024)
    return Sb, prob


def cal_si(num_each_class, ss_dict, mu_i):
    sub_si = {}
    for i in range(NUM_Labels):
        j = i + 1
        mu_sub = np.array(mu_i[i])
        mu_sub = mu_sub.reshape(1,-1)
        x_sub = ss_dict[j]
        sub = np.dot((x_sub-mu_sub).T,(x_sub-mu_sub))/num_each_class[i][1]
        sub_si[j] = sub

    # print(sub_si[1].shape)
    return sub_si


def cal_sw(prob, Si):
    Sw = 0
    for i in range(NUM_Labels):
        Sw += prob[i] * Si[i+1]
    # print(Sw.shape)                                                   # shape: (1024,1024)
    return Sw


def LDA_Transform(x_mat, selectVec):
    trans_data = np.dot(x_mat, selectVec.T)
    return trans_data


def LDA(x_train, num_class, num_each_class, k):
    mu_sum = cal_mu_sum(x_train)
    ss_dict = compute_ss(x_train, y_train)                              # format: {1:array[]}
    mu_i = cal_mu_i(ss_dict, num_class)                                 # format: [[],[],[]]; len = 20
    Sb, prob = cal_sb(num_each_class, mu_i, mu_sum)
    print('Probability of each class:',prob)
    Si = cal_si(num_each_class, ss_dict, mu_i)                          # format: {1:(1024,1024),2:(1024,1024)}
    Sw = cal_sw(prob, Si)
    matrix = np.dot(np.linalg.pinv(Sw), Sb)
    featValue, featVec = np.linalg.eig(matrix)
    index = np.argsort(-featValue)
    selectVec = np.array(featVec.T[index[:k]])  # pay attention to transpose
    trans_data = np.dot(x_train, selectVec.T)
    return trans_data, selectVec


def show_graph(k, acc):
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    train_acc = acc[2]['train_acc']
    test_acc = acc[2]['test_acc']
    ax1.plot(k, train_acc, label='train')
    ax1.plot(k, test_acc, label='test')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy Rates')
    plt.title('LDA with 2-dimension')
    ax1.legend()

    ax2 = fig.add_subplot(132)
    train_acc = acc[3]['train_acc']
    test_acc = acc[3]['test_acc']
    ax2.plot(k, train_acc, label='train')
    ax2.plot(k, test_acc, label='test')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy Rates')
    plt.title('LDA with 3-dimension')
    ax2.legend()

    ax3 = fig.add_subplot(133)
    train_acc = acc[9]['train_acc']
    test_acc = acc[9]['test_acc']
    ax3.plot(k, train_acc, label='train')
    ax3.plot(k, test_acc, label='test')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy Rates')
    plt.title('LDA with 9-dimension')
    ax3.legend()

    plt.show()


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


if __name__ == '__main__':
    random.seed(100)
    global NUM_Labels
    global sum_pics
    sum_pics = 2380
    NUM_Labels = 20

    x_train, x_test, y_train, y_test = dataLoading()
    DIM = [2, 3, 9]
    k_number = [1, 3, 5]
    num_class, num_each_class = cal_label(y_train)
    acc = {}
    for i in DIM:
        acc[i] = {}
        train_acc_list = []
        test_acc_list = []
        print('------------------------')
        print('When dimention is:', i)
        train_transData, selectVec = LDA(x_train, num_class, num_each_class, i)
        test_transData = LDA_Transform(x_test, selectVec)
        for j in k_number:
            print('When consider', j, 'neighbors:')
            # Calculate training accuracy
            KNN = knn_classifier(j)
            train_acc = KNN.clf(train_transData, y_train)
            train_acc_list.append(train_acc)
            print('Training accuracy:', train_acc)
            # Calculate test accuracy
            test_acc = KNN.pred_clf(train_transData, y_train, test_transData, y_test)
            test_acc_list.append(test_acc)
            print('Testing accuracy:', test_acc)
        acc[i]['train_acc'] = train_acc_list
        acc[i]['test_acc'] = test_acc_list
    print(acc)
    show_graph(k_number, acc)
