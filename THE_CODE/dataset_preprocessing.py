
######## IMPORTS #########
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #hide info from CPU optimizer and Deep Neural Network Library 

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import extract_data
#from sklearn.model_selection import train_test_split

######## VARIABLES ############

std_awgn = 2 #standard deviation for awgn addition
train_test_ratio = 0.1

######## DATA LOADING #########

dataset = extract_data.load_data("MPA-MLF_data\\Train", (72, 48))
dataset_labels = extract_data.load_labels("MPA-MLF_data\\label_train.csv")

####### DATA AUGMENTATION ########

dataset, dataset_labels = extract_data.data_augm(dataset, dataset_labels, std_awgn)

print("Final size of dataset: ", dataset.shape)

######## DATA PREPROCESSING #########

x_train, y_train, x_test,  y_test = extract_data.split_data(dataset, dataset_labels, train_test_ratio)

print("Num of bts0, bts1, bts2 in training data: ", np.count_nonzero(y_train == 0), np.count_nonzero(y_train == 1), np.count_nonzero(y_train == 2), "num of train: ", len(y_train))
print("Num of bts0, bts1, bts2 in test data: ", np.count_nonzero(y_test == 0), np.count_nonzero(y_test == 1), np.count_nonzero(y_test == 2), "num of test: ", len(y_test))

y_train_encoded = to_categorical(y_train, num_classes = 3)
y_test_encoded = to_categorical(y_test, num_classes = 3)

x_train_mean = np.mean(x_train) #Z-score normalization of training data
x_train_deviation = np.std(x_train)

np.save("aux_data_pool\\mean_dev.npy", np.array([x_train_mean, x_train_deviation]))
x_train_normalized = (x_train - x_train_mean) / x_train_deviation

x_test_normalized = (x_test - x_train_mean) / x_train_deviation

x_train_normalized = x_train_normalized.reshape(-1, 72, 48,1)
x_test_normalized = x_test_normalized.reshape(-1, 72, 48, 1)

np.savez_compressed("aux_data_pool\\train_data.npz", x = x_train_normalized, y = y_train_encoded)#
np.savez_compressed("aux_data_pool\\test_data.npz", x = x_test_normalized, y = y_test_encoded)

