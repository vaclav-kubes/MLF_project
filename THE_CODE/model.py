
######## IMPORTS #########

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, BatchNormalization
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import extract_data

######## VARIABLES ############
SAVE_MODEL = True

######## DATA LOADING #########

dataset = extract_data.load_data("MPA-MLF_data\\Train", (72, 48))
dataset_labels = extract_data.load_labels("MPA-MLF_data\\label_train.csv")
#print(np.count_nonzero(y_train == 0),np.count_nonzero(y_train == 1), np.count_nonzero(y_train == 2) )
#x_test = extract_data.load_data("MPA-MLF_data\\Test", (72, 48))
#y_test = extract_data.load_labels("MPA-MLF_data\\test_format.csv")
#rnd_test_index_list = np.random.choice(range(0,x_train.shape[0]), 500)
#print(rnd_test_index_list)
#x_test = x_train[rnd_test_index_list]
#y_test = y_train[rnd_test_index_list]
#print(y_test)
print(np.std(dataset))
####### DATA AUGMENTATION ########
bts_1 = dataset[dataset_labels == 1]
print("Num of bts 1: ", bts_1.shape[0])
num_of_bts0 = np.count_nonzero(dataset_labels == 0)
for k in range(num_of_bts0//bts_1.shape[0]):
    bts_1 = bts_1 + np.random.normal(0, 3, (bts_1.shape[0], bts_1.shape[1], bts_1.shape[2])) #adding AWGN
    dataset = np.append(dataset, bts_1, 0)
    dataset_labels = np.append(dataset_labels, np.ones(bts_1.shape[0]))

bts_2 = dataset[dataset_labels == 2]
print("Num of bts 2: ", bts_2.shape[0])
for k in range(num_of_bts0//bts_2.shape[0]):
    bts_2 = bts_2 + np.random.normal(0, 3, (bts_2.shape[0], bts_2.shape[1], bts_2.shape[2])) #adding AWGN
    dataset = np.append(dataset, bts_2, 0)
    dataset_labels = np.append(dataset_labels, 2 * np.ones(bts_2.shape[0]))

print("Final size of dataset: ", dataset.shape)

######## DATA PREPROCESSING #########

print(np.max(dataset), np.min(dataset))

x_train, y_train, x_test, y_test = extract_data.split_data(dataset, dataset_labels, 0.3)

print("Num of bts0, bts1, bts2 in training data: ", np.count_nonzero(y_train == 0), np.count_nonzero(y_train == 1), np.count_nonzero(y_train == 2), "num of train: ", len(y_train))
print("Num of bts0, bts1, bts2 in test data: ", np.count_nonzero(y_test == 0), np.count_nonzero(y_test == 1), np.count_nonzero(y_test == 2), "num of test: ", len(y_test))

y_train_encoded = to_categorical(y_train, num_classes=3)
y_test_encoded = to_categorical(y_test, num_classes=3)

x_train_mean = np.mean(x_train) #Z-score normalization of training data
x_train_deviation = np.std(x_train)
x_train_normalized = (x_train - x_train_mean) / x_train_deviation

x_test_mean = np.mean(x_test) #Z-score normalization of test data
x_test_deviation = np.std(x_test)
x_test_normalized = (x_test - x_test_mean) / x_test_deviation

#print(x_train_normalized.shape)
x_train_normalized = x_train_normalized.reshape(-1, 72, 48,1)
x_test_normalized = x_test_normalized.reshape(-1, 72, 48, 1)
#print(x_train_normalized.shape)
######## CNN MODEL #########

model = Sequential()
#model.add(Input(x_train_normalized.shape))
model.add(Input(shape=(72, 48, 1)))
model.add(Conv2D(64, kernel_size=(4,4), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten(input_shape=(32, 32, 2)))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(3, activation='softmax'))


optimizer = Adam(learning_rate = 0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.summary()


early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True  )

######## TRAINING #########
#if LOAD_SAVED_MODEL:

#else:
history = model.fit(x_train_normalized, y_train_encoded, epochs=30, batch_size=20, validation_split = 0.2, shuffle = True, callbacks = early_stopping)#,

if SAVE_MODEL: model.save("THE_CODE\\model.keras")


######## EVALUATION #########

score = model.evaluate(x_test_normalized, y_test_encoded, verbose=1)
print('Test loss:', score[0])
print(f'Test accuracy: {score[1]*100} %')


#extract_data.load_data("MPA-MLF_data\Test", (72, 48))

plt.figure()
plt.title('Loss')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.grid('both')
plt.figure()
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
plt.show()
