import numpy as np
import torch

def modify_cifar(dataset):
    # remove frogs and shift everything after
    dataset.train_labels = np.asarray(dataset.train_labels)
    dataset.train_data = np.delete(dataset.train_data, np.where(dataset.train_labels==6)[0], axis=0)
    dataset.train_labels = np.delete(dataset.train_labels, np.where(dataset.train_labels==6)[0], axis=0)
    dataset.train_labels = np.asarray([x-1 if x>= 7 else x for x in  dataset.train_labels])
    # exchange automobiles and birds
    idx1 = np.where(dataset.train_labels == 1)
    idx2 = np.where(dataset.train_labels == 2)
    dataset.train_labels[idx1] = 2
    dataset.train_labels[idx2] = 1
    
def modify_cifar_t(dataset):
    # remove frogs and shift everything after
    dataset.test_labels = np.asarray(dataset.test_labels)
    dataset.test_data = np.delete(dataset.test_data, np.where(dataset.test_labels==6)[0], axis=0)
    dataset.test_labels = np.delete(dataset.test_labels, np.where(dataset.test_labels==6)[0], axis=0)
    dataset.test_labels = np.asarray([x-1 if x>= 7 else x for x in  dataset.test_labels])
    # exchange automobiles and birds
    idx1 = np.where(dataset.test_labels == 1)
    idx2 = np.where(dataset.test_labels == 2)
    dataset.test_labels[idx1] = 2
    dataset.test_labels[idx2] = 1

def modify_stl(dataset):
    # remove monkeys and shift everything after
    dataset.labels = np.asarray(dataset.labels)
    dataset.data = np.delete(dataset.data, np.where(dataset.labels==7)[0], axis=0)
    dataset.labels = np.delete(dataset.labels, np.where(dataset.labels == 7)[0], axis=0)
    dataset.labels = np.asarray([x - 1 if x >= 8 else x for x in dataset.labels])
