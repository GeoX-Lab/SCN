import random

import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import os
from torch.utils.data import random_split
import glob
from PIL import Image
import numpy as np
from libtiff import TIFF


def find_classes(class_file):
    classes = os.listdir(class_file)

    classes = sorted(classes)
    classes.sort()
    classes_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, classes_to_idx


def make_dataset(root, base_folder, train, train_test_rate, class_to_idx):
    test_img_path = []
    train_img_path = []
    test_labels = []
    train_labels = []
    dir_path = os.path.join(root, base_folder)
    # get dir path
    cat_dirs = os.listdir(dir_path)
    for fdir in cat_dirs:
        img_path = []
        labels = []
        all_figure = glob.glob(os.path.join(dir_path, fdir, '*.jpg'))
        # get all pictures path
        for fimg in all_figure:
            img_path.append(fimg)
            labels.append(class_to_idx[fdir])

        train_img, test_img, train_target, test_target = train_test_split(img_path, labels,
                                                                          test_size=1 - train_test_rate)

        train_img_path += train_img
        train_labels += train_target
        test_img_path += test_img
        test_labels += test_target
    if train:
        return train_img_path, train_labels
    else:
        return test_img_path, test_labels


class NWPU():
    base_folder = ''
    name = 'NWPU'
    class_num = 45

    def __init__(self, root, train=True, train_test_rate=0.8):
        self.root = os.path.expanduser(root)
        self.train = train
        self.train_test_rate = train_test_rate
        _, class_to_idx = find_classes(os.path.join(self.root, self.base_folder))
        self.data, self.targets = make_dataset(self.root,
                                               self.base_folder,
                                               self.train,
                                               self.train_test_rate,
                                               class_to_idx)


class iNWPU(NWPU):
    def __init__(self, root, train,
                 test_transform=None,
                 train_transform=None):
        super(iNWPU, self).__init__(root, train)
        self.ucmerced = NWPU(root, train)
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []

    def concatenate(self, datas, labels):
        con_data = datas[0]
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label, labels[i]), axis=0)
        return con_data, con_label

    def getTestData(self, classes):
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            self.data = np.array(self.data)
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        # datas,labels=self.concatenate(datas,labels)
        # self.TestData=datas if self.TestData==[] else np.concatenate((self.TestData,datas),axis=0)
        # self.TestLabels=labels if self.TestLabels==[] else np.concatenate((self.TestLabels,labels),axis=0)
        self.TestData, self.TestLabels = self.concatenate(datas, labels)
        print("the size of test set is %s" % (str(self.TestData.shape)))
        print("the size of test label is %s" % str(self.TestLabels.shape))

    def getTrainData(self, classes, exemplar_set):
        datas, labels = [], []
        if len(exemplar_set) != 0:
            datas = [exemplar for exemplar in exemplar_set]
            length = len(datas[0])
            labels = [np.full((length), label) for label in range(len(exemplar_set))]

        for label in range(classes[0], classes[1]):
            self.data = np.array(self.data)
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)
        print("the size of train set is %s" % (str(self.TrainData.shape)))
        print("the size of train label is %s" % str(self.TrainLabels.shape))

    def getTrainItem(self, index):
        # img = loadTifImage(self.TrainData[index])
        img = cv2.imread(self.TrainData[index])

        img = Image.fromarray(img)
        target = self.TrainLabels[index]

        if self.train_transform:
            img = self.train_transform(img)

        # if self.target_transform:
        #     target = self.target_transform(target)

        return index, img, target

    def getTestItem(self, index):
        # img = loadTifImage(self.TrainData[index])
        img = cv2.imread(self.TestData[index])
        img = Image.fromarray(img)
        target = self.TestLabels[index]

        if self.test_transform:
            img = self.test_transform(img)

        # if self.target_test_transform:
        #     target = self.target_test_transform(target)

        return index, img, target

    def __getitem__(self, index):
        if len(self.TrainData) is not 0:
            return self.getTrainItem(index)
        elif len(self.TestData) is not 0:
            return self.getTestItem(index)

    def __len__(self):
        if len(self.TrainData) is not 0:
            return len(self.TrainData)
        elif len(self.TestData) is not 0:
            return len(self.TestData)

    def get_image_class(self, label):
        return self.data[np.array(self.targets) == label]
