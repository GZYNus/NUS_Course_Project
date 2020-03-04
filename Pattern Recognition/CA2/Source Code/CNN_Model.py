import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
import matplotlib.pyplot as plt


class ConvNet(nn.Module):
    """
    Define a Convolutional Neural Network
    """
    def __init__(self, num_class):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(20, 50, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Linear(50*5*5, 500),
            # nn.ReLU(),
            # nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, num_class)
        )

    # forward propagation
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.layer3(out)
        return out


def myLoad():
    my_dataset = np.fromfile('my_array.bin', dtype=np.float64)
    my_label = np.fromfile('my_label.bin',dtype=int)
    my_dataset = my_dataset.reshape(10,-1)
    return my_dataset, my_label


class TrainDataSet:
    """
    Data loader for training data
    """

    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __getitem__(self, item):
        input_tensor = self.inputs[item, :]
        label_tensor = self.labels[item]

        return input_tensor, label_tensor

    def __len__(self):
        return self.inputs.shape[0]


def dataloding():
    image_dataset = np.fromfile('image_array.bin', dtype=np.float64)
    label = np.fromfile('label.bin', dtype=int)
    image_dataset = image_dataset.reshape(3400, -1)
    return image_dataset, label


def Training(EPOCH, trainloader, cnn_model, batch_size, criterion, optimizer):
    CNN_MODEL = cnn_model
    for e in range(EPOCH):
        print('Epoch:',e)
        for step, DATALOADER in enumerate(trainloader):
            x_sub_tr = DATALOADER[0]
            y_sub_tr = DATALOADER[1]
            x_sub_tr = x_sub_tr.type(torch.FloatTensor)
            y_sub_tr = y_sub_tr.type(torch.LongTensor)
            x_sub_tr = x_sub_tr.view(batch_size, 1, 32, 32)
            CNN_MODEL.zero_grad()
            yhat = CNN_MODEL(x_sub_tr)
            y_sub_tr = y_sub_tr - 1             # When using CrossEntropy, class label must start from 0
            loss = criterion(yhat, y_sub_tr)
            # print('loss:',loss)
            loss.backward()             # BP: back propagation
            optimizer.step()
    return CNN_MODEL


def test(x_test, trainedModel, y_test):
    x_test = x_test.reshape(-1, 1, 32, 32)
    trainset = TrainDataSet(x_test, y_test)
    trainloader = Data.DataLoader(
        dataset=trainset,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=16)
    acc = 0
    for step, DATALOADER in enumerate(trainloader):
        x_sub_tes = DATALOADER[0].type(torch.FloatTensor)
        y_sub_tes = DATALOADER[1].type(torch.int)
        yhat = trainedModel(x_sub_tes)
        clf_test = F.softmax(yhat, dim=1)
        index = clf_test.argmax(dim=1) +1
        index = index.type(torch.int)
        for i in range(len(index)):
            if index[i] == y_sub_tes[i]:
                acc += 1
    return acc


def show_acc(acc):
    EPOCH = [200,400,600]
    plt.plot(EPOCH, acc, label='Accuracy')
    plt.xlabel('EPOCH')
    plt.ylabel('Accuracy Rates')
    plt.title('Accuracy with different epoch')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    image_dataset, label = dataloding()
    x_train, x_test, y_train, y_test = train_test_split(image_dataset, label, test_size=0.3, random_state=100)
    my_pic, my_label = myLoad()
    y_train = np.concatenate((y_train, my_label[3:]), axis=0)
    x_train = np.concatenate((x_train, my_pic[3:]), axis=0)
    x_test = np.concatenate((x_test, my_pic[:3]), axis=0)
    y_test = np.concatenate((y_test, my_label[:3]), axis=0)
    # Establish 3 models with different epochs(iterations)
    EPOCH = [5, 10, 15]
    lr = 10e-3
    batch_size = 64
    # Model initialization
    cnn_model = ConvNet(21)
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropy Loss function
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=lr)
    trainset = TrainDataSet(x_train, y_train)
    trainloader = Data.DataLoader(
        dataset=trainset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=16)
    acc_list = []
    for i in EPOCH:
        trainedModel = Training(i, trainloader, cnn_model, batch_size, criterion, optimizer)
        acc = test(x_test, trainedModel, y_test)
        acc = acc/len(y_test)
        acc_list.append(acc)
    print(acc_list)
    show_acc(acc_list)






