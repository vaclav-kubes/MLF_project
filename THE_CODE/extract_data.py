
import os
import numpy as np

def load_data(directory, inp_size):
    #directory = "D:\\Users\\User\\Documents\\MLF_project\\MPA-MLF_data\\Train"
    files = os.listdir(directory)
    files.sort()

    train_data = np.zeros((len(files), inp_size[0], inp_size[2]))

    for file in files:
        if ".npy" in file:
            path = os.path.join(directory, file)
            train_data[files.index(file)] = np.load(path, "r")

    return train_data

def load_labels(path):
    labels = np.genfromtxt(path, delimiter=",")
    return labels[1:, 1]
