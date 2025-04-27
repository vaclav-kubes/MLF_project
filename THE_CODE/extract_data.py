
import os
import numpy as np

def load_data(directory, inp_size):
    #directory = "D:\\Users\\User\\Documents\\MLF_project\\MPA-MLF_data\\Train"
    files = []
    for f in os.listdir(directory): 
        if ".npy" in f: files.append(int(f[:-4]))
    files.sort()
    #print(files)
    train_data = np.zeros((len(files), inp_size[0], inp_size[1]))

    for file in files:
        path = os.path.join(directory, str(file) + ".npy")
        train_data[files.index(file)] = np.load(path, "r")

    return train_data

def load_labels(path):
    labels = np.genfromtxt(path, delimiter=",")
    return labels[1:, 1]


def split_data(dataset: np.array, dataset_labels: np.array,  train_eval_ratio: float) -> tuple: 
    num_of_eval = np.int64(np.ceil(train_eval_ratio*dataset.shape[0]))
    print(num_of_eval)
    np.random.seed(0)
    np.random.shuffle(dataset)
    np.random.seed(0)
    np.random.shuffle(dataset_labels)
    
    rnd_test_index_list = np.random.choice(range(dataset.shape[0]), num_of_eval, replace = False)
    #rnd_test_index_list = np.random.randint(0, dataset.shape[0], num_of_eval)
    eval_labels = dataset_labels[rnd_test_index_list]
    eval = dataset[rnd_test_index_list]
    train = [dataset[k] for k in range(0,dataset.shape[0] - 1) if k not in rnd_test_index_list]
    train_labels = np.array([dataset_labels[k] for k in range(0,dataset_labels.shape[0] - 1) if k not in rnd_test_index_list])
    
    #train = dataset[train_index_list]
    return train, train_labels , eval, eval_labels

def data_augm(dataset: np.array, dataset_labels: np.array, awgn_std: int) -> tuple:
    bts_0 = dataset[dataset_labels == 0]
    bts_1 = dataset[dataset_labels == 1]
    bts_2 = dataset[dataset_labels == 2]
    #mean = np.mean(dataset)
    for k in range(2):
        bts_0 = bts_0 + np.random.normal(0, awgn_std + 2*k, (bts_0.shape[0], bts_0.shape[1], bts_0.shape[2])) #adding AWGN
        dataset = np.append(dataset, bts_0, 0)
        dataset_labels = np.append(dataset_labels, np.zeros(bts_0.shape[0]))

    num_of_bts0 = np.count_nonzero(dataset_labels == 0)

    for k in range(num_of_bts0//bts_1.shape[0]):
        bts_1 = bts_1 + np.random.normal(0, awgn_std + k//3, (bts_1.shape[0], bts_1.shape[1], bts_1.shape[2])) #adding AWGN
        dataset = np.append(dataset, bts_1, 0)
        dataset_labels = np.append(dataset_labels, np.ones(bts_1.shape[0]))

    for k in range(num_of_bts0//bts_2.shape[0] ):
        bts_2 = bts_2 + np.random.normal(0, awgn_std + k//3, (bts_2.shape[0], bts_2.shape[1], bts_2.shape[2])) #adding AWGN
        dataset = np.append(dataset, bts_2, 0)
        dataset_labels = np.append(dataset_labels, 2 * np.ones(bts_2.shape[0]))

    return dataset, dataset_labels