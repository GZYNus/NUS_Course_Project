import random
import matplotlib.pyplot as plt
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import os


# Store all images and labels to bin files
def dataPreparing():
    label = []
    for k in range(1,21):
        for i in range(1,171):
            I = cv2.imread('./PIE/%d/%d.jpg'%(k,i), 0)
            if (i == 1) and (k == 1):
                label.append(k)
                image_dataset = I
            else:
                label.append(k)
                image_dataset = np.concatenate((image_dataset, I), axis=0)

    image_dataset = image_dataset.astype("float") / 255.0
    image_dataset.tofile('image_array.bin')
    label = np.array(label)
    label.tofile('label.bin')


# my 10 photos to bin files
def my_pht_process():
    label = []
    for i in range(1, 11):
        I = cv2.imread('./my_pht/%d.jpg' % i, 0)
        I = I.reshape(1, -1)
        label.append(21)
        if i == 1:
            my_dataset = I
        else:
            my_dataset = np.concatenate((my_dataset, I), axis=0)
    my_dataset = my_dataset.astype("float") / 255.0
    my_dataset.tofile('my_array.bin')
    label = np.array(label)
    label.tofile('my_label.bin')


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


# Create PCA model
def pca(XMat, k):
    average = np.mean(XMat,0)
    m, n = np.shape(XMat)
    # avgs = np.tile(average, (m, 1))
    data_adjust = XMat - average
    covX = np.cov(data_adjust.T)   # cal covariance matrix
    featValue, featVec = np.linalg.eig(covX)    # cal eigenvalue & eigenvector
    index = np.argsort(-featValue)  # descending sort
    if k > n:
        print("k must be smaller than feature number")
        return
    else:
        selectVec = np.array(featVec.T[index[:k]])  # pay attention to transpose
        finalData = np.dot(data_adjust, selectVec.T)
    return finalData, selectVec


# Visualization 2D
def visualization_2PC(transData, y_train):
    plt.scatter(transData[:, 0].real, transData[:, 1].real, c=y_train, edgecolor='none',
                alpha=0.5, cmap=plt.cm.get_cmap('Spectral', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.show()


# Visualization 3D
def visualization_3PC(transData, y_train):
    x = transData[:, 0].real
    y = transData[:, 1].real
    z = transData[:, 2].real
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, c=y_train)
    plt.show()


def show_eigface(selectVec):
    fig = plt.figure()
    selectVec = selectVec.real
    plt.subplot(131)
    face1 = selectVec[0].reshape(32,32) * 255
    plt.imshow(face1, cmap='Greys_r')
    # plt.show
    plt.subplot(132)
    face2 = selectVec[1].reshape(32,32) * 255
    plt.imshow(face2, cmap='Greys_r')
    # plt.show()
    plt.subplot(133)
    face3 = selectVec[2].reshape(32, 32) * 255
    plt.imshow(face3, cmap='Greys_r')
    plt.show()


if __name__ == '__main__':
    work_path = os.getcwd()
    random.seed(100)
    # dataPreparing()
    # my_pht_process()
    x_train, y_train = dataLoading()
    my_train, my_label = myLoad()
    y_train = np.concatenate((y_train, my_label), axis=0)
    x_train = np.concatenate((x_train, my_train), axis=0)
    # Visualization when PCs equal to 2
    transData, selectVec = pca(x_train, 2)
    visualization_2PC(transData, y_train)
    # Visualization when PCs equal to 3
    transData, selectVec = pca(x_train, 3)
    visualization_3PC(transData, y_train)
    show_eigface(selectVec)








