
######## IMPORTS #########

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import extract_data


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


######## DATA PREPROCESSING #########

print(np.max(dataset), np.min(dataset))

x_train, y_train, x_test, y_test = extract_data.split_data(dataset, dataset_labels, 0.2)

y_train_encoded = to_categorical(y_train, num_classes=3)
y_test_encoded = to_categorical(y_test, num_classes=3)

x_train_mean = np.mean(x_train) #Z-score normalization of training data
x_train_deviation = np.std(x_train)
x_train_normalized = (x_train - x_train_mean) / x_train_deviation

x_test_mean = np.mean(x_test) #Z-score normalization of test data
x_test_deviation = np.std(x_test)
x_test_normalized = (x_test - x_test_mean) / x_test_deviation

x_train_normalized = x_train_normalized.reshape(-1, 72, 48,1)
x_test_normalized = x_test_normalized.reshape(-1, 72, 48, 1)

######## CNN MODEL #########

model = Sequential()
model.add(Input(shape=(72, 48, 1)))
model.add(Conv2D(32, kernel_size=(3,3), activation = 'relu', input_shape=(72,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten(input_shape=(32, 32, 2)))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))


optimizer = Adam(learning_rate = 0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.summary()


early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True  )

######## TRAINING #########

history = model.fit(x_train_normalized, y_train_encoded, epochs=30, batch_size=10, validation_split = 0.2, callbacks=early_stopping)


######## EVALUATION #########

score = model.evaluate(x_test_normalized, y_test_encoded, verbose=1)
print('Test loss:', score[0])
print(f'Test accuracy: {score[1]*100} %')


extract_data.load_data("MPA-MLF_data\Test", (72, 48))

y_pred = model.predict(x_test_normalized)
print(y_pred)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test_encoded, axis=1)
print(y_pred_classes)
print(y_true_classes)
cm = confusion_matrix(y_true_classes, y_pred_classes)
print(cm)
ConfusionMatrixDisplay.from_predictions(y_true_classes,y_pred_classes)

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
plt.show()