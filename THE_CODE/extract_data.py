
import os
import numpy as np

def load_data(directory, inp_size):
    #directory = "D:\\Users\\User\\Documents\\MLF_project\\MPA-MLF_data\\Train"
    files = os.listdir(directory)
    files.sort()

    train_data = np.zeros((len(files), inp_size[0], inp_size[1]))

    for file in files:
        if ".npy" in file:
            path = os.path.join(directory, file)
            train_data[files.index(file)] = np.load(path, "r")

    return train_data

def load_labels(path):
    labels = np.genfromtxt(path, delimiter=",")
    return labels[1:, 1]


def split_data(dataset: np.array, dataset_labels: np.array,  train_eval_ratio: float) -> tuple: 
    num_of_eval =np.int64(np.ceil(train_eval_ratio*dataset.shape[0]))
    print(num_of_eval)
    #rnd_test_index_list = np.random.choice(range(0,dataset.shape[0] - 1), num_of_eval)
    rnd_test_index_list = np.random.randint(0, dataset.shape[0], num_of_eval)
    eval_labels = dataset_labels[rnd_test_index_list]
    eval = dataset[rnd_test_index_list]
    train = [dataset[k] for k in range(0,dataset.shape[0] - 1) if k not in rnd_test_index_list]
    train_labels = np.array([dataset_labels[k] for k in range(0,dataset_labels.shape[0] - 1) if k not in rnd_test_index_list])
    
    #train = dataset[train_index_list]
    return train, train_labels , eval, eval_labels