from svmutil import *
from svm import *
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D


# load images and labels and generate training set, including randomly selected 500 images and labels
def dataLoading():
    image_dataset = np.fromfile('image_array.bin',dtype=np.float64)
    label = np.fromfile('label.bin', dtype= int)
    image_dataset = image_dataset.reshape(3400,-1)              # (3400, 1024)
    random_index = [random.randint(0, 3399) for i in range(500)]
    x_train = image_dataset[random_index]
    y_train = label[random_index]
    return x_train, y_train


def myLoad():
    my_dataset = np.fromfile('my_array.bin', dtype=np.float64)
    my_label = np.fromfile('my_label.bin',dtype=int)
    my_dataset = my_dataset.reshape(10,-1)
    return my_dataset, my_label


def norm(x_mat):
    return x_mat-np.mean(x_mat,0)


def cal_mu_sum(x_mat):
    return np.mean(x_mat,0)


def compute_ss(x_train, y_train):
    ss = {}
    for i in range(NUM_Label):
        i += 1
        ss[i] = []
    for i in range(510):
        a = y_train[i]
        ss[a].append(x_train[i])
    for i in range(NUM_Label):
        i += 1
        ss_sub = np.array(ss[i])
        ss[i] = ss_sub
    return ss


def cal_mu_i(ss, num_class):
    mu_i = []
    for i in range(num_class):
        i += 1
        a = np.mean(ss[i], 0)
        mu_i.append(a)
    return mu_i


def cal_sb(num_each_class, mu_i, mu_sum):
    sb_i = {}
    prob = []
    for i in range(NUM_Label):
        prob.append(num_each_class[i][1]/510)

    for i in range(NUM_Label):
        j = i + 1
        aa = np.array(mu_i[i]-mu_sum)
        aa = aa.reshape(1,-1)
        sb_i[j] = prob[i] * np.dot(aa.T,aa)

    Sb = 0
    for i in range(NUM_Label):
        j = i + 1
        Sb += sb_i[j]
    # print(Sb.shape)                                                   # shape: (1024,1024)
    return Sb, prob


def cal_si(num_each_class, ss_dict, mu_i):
    sub_si = {}
    for i in range(NUM_Label):
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
    for i in range(NUM_Label):
        Sw += prob[i] * Si[i+1]
    # print(Sw.shape)                                                   # shape: (1024,1024)
    return Sw


def LDA(x_train, num_class, num_each_class, k):
    mu_sum = cal_mu_sum(x_train)
    ss_dict = compute_ss(x_train, y_train)                              # format: {1:array[]}
    mu_i = cal_mu_i(ss_dict, num_class)                                 # format: [[],[],[]]; len = 20
    # print(len(mu_i[0]))
    Sb, prob = cal_sb(num_each_class, mu_i, mu_sum)
    Si = cal_si(num_each_class, ss_dict, mu_i)                          # format: {1:(1024,1024),2:(1024,1024)}
    Sw = cal_sw(prob, Si)
    # matrix = np.dot(np.linalg.inv(Sw), Sb)
    matrix = np.dot(np.linalg.pinv(Sw),Sb)
    featValue, featVec = np.linalg.eig(matrix)
    index = np.argsort(-featValue)
    selectVec = np.array(featVec.T[index[:k]])  # pay attention to transpose
    trans_data = np.dot(x_train, selectVec.T)
    return trans_data


def cal_label(y_train):
    classes = list(set(y_train))
    num_class = len(classes)                                            # int: 20
    num_each_class = dict(Counter(y_train))
    num_each_class = sorted(num_each_class.items(), key=lambda x:x[0])   # format: [(1, 26), (2, 27), (3, 34)]
    return num_class, num_each_class


# Visualization 2D
def visualization_2PC(transData):
    plt.scatter(transData[:, 0].real, transData[:, 1].real, edgecolor='none', c=y_train,
                alpha=0.5, cmap=plt.cm.get_cmap('Spectral', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.show()


# Visualization 3D
def visualization_3PC(transData):
    x = transData[:, 0].real
    y = transData[:, 1].real
    z = transData[:, 2].real
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, c=y_train)
    plt.show()


if __name__ == '__main__':
    random.seed(100)
    global NUM_Label
    NUM_Label = 21
    x_train, y_train = dataLoading()
    # x_train = x_train * 255

    my_train, my_label = myLoad()
    y_train = np.concatenate((y_train, my_label), axis=0)
    x_train = np.concatenate((x_train, my_train), axis=0)
    # x_train = norm(x_train)
    num_class, num_each_class = cal_label(y_train)
    # Visualization when PCs are 2
    trans_data = LDA(x_train, num_class, num_each_class, 2)
    visualization_2PC(trans_data)
    # Visualization when PCs are 3
    trans_data = LDA(x_train, num_class, num_each_class, 3)
    visualization_3PC(trans_data)


