
######## IMPORTS #########

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.optimizers import Adam, AdamW
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, BatchNormalization, AveragePooling2D
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import extract_data
#from sklearn.model_selection import train_test_split

######## VARIABLES ############

SAVE_MODEL = True
std_awgn = 2 #standard deviation for awgn addition
train_test_ratio = 0.2

######## DATA LOADING #########

dataset = extract_data.load_data("MPA-MLF_data\\Train", (72, 48))
dataset_labels = extract_data.load_labels("MPA-MLF_data\\label_train.csv")
#(print(np.count_nonzero(dataset_labels == 0),np.count_nonzero(dataset_labels == 1), np.count_nonzero(dataset_labels == 2) )
#x_test = extract_data.load_data("MPA-MLF_data\\Test", (72, 48))
#y_test = extract_data.load_labels("MPA-MLF_data\\test_format.csv")
#rnd_test_index_list = np.random.choice(range(0,x_train.shape[0]), 500)
#print(rnd_test_index_list)
#x_test = x_train[rnd_test_index_list]
#y_test = y_train[rnd_test_index_list]
#print(y_test)
#print(np.std(dataset))

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
#mean = np.mean(dataset)
#deviation = np.std(dataset)
np.save("THE_CODE\\mean_dev.npy",np.array([x_train_mean, x_train_deviation]))
x_train_normalized = (x_train - x_train_mean) / x_train_deviation

#x_test_mean = np.mean(x_test) #Z-score normalization of test data
#x_test_deviation = np.std(x_test)
x_test_normalized = (x_test - x_train_mean) / x_train_deviation

#print(x_train_normalized.shape)
x_train_normalized = x_train_normalized.reshape(-1, 72, 48,1)
x_test_normalized = x_test_normalized.reshape(-1, 72, 48, 1)
#print(x_train_normalized.shape)

######## CNN MODEL #########

model = Sequential()
#model.add(Input(x_train_normalized.shape))
model.add(Input(shape=(72, 48, 1)))
model.add(Conv2D(32, kernel_size=(5,5), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten()) #input_shape=(32, 32, 2)
#model.add(BatchNormalization())
model.add(Dense(64, activation='relu')) #128
#model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
#model.add(BatchNormalization())
model.add(Dense(3, activation='softmax'))

optimizer = Adam(learning_rate = 0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=7, verbose=1, restore_best_weights=True)

######## TRAINING #########

#if LOAD_SAVED_MODEL:

#else:                                                                                                 #0.5
history = model.fit(x_train_normalized, y_train_encoded, epochs=30, batch_size=20, validation_data = (x_test_normalized, y_test_encoded), callbacks = early_stopping)#validation_split = 0.2, shuffle = True

if SAVE_MODEL: model.save("THE_CODE\\model.keras")


######## EVALUATION #########

score = model.evaluate(x_test_normalized, y_test_encoded, verbose=1)
print('Test loss:', score[0])
print(f'Test accuracy: {score[1]*100} %')

plt.figure()
plt.subplot(2, 1, 1)
plt.title('Loss')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.grid('both')

plt.subplot(2, 1, 2)
plt.title('Accuracy')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.grid('both')

y_pred = model.predict(x_test_normalized)
#print(y_pred)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test_encoded, axis=1)
#print(y_pred_classes)
#print(y_true_classes)
cm = confusion_matrix(y_true_classes, y_pred_classes)
print(cm)
ConfusionMatrixDisplay.from_predictions(y_true_classes,y_pred_classes)

unknown_data = extract_data.load_data("MPA-MLF_data\\Test", (72, 48))
#mean = np.mean(unknown_data) #Z-score normalization of training data
#deviation = np.std(unknown_data)
unknown_data_norm = (unknown_data - x_train_mean) / x_train_deviation
pred = model.predict(unknown_data_norm.reshape(-1, 72, 48, 1))
array_to_save = np.vstack((np.arange(0, 119 + 1), np.argmax(pred, axis=1))).astype(np.int16)
np.savetxt("THE_CODE\\my_guess.csv", array_to_save.transpose(),"%d", ",", header="ID,target", comments="")

plt.show()