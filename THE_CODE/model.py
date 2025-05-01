
######## IMPORTS #########

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.optimizers import Adam, AdamW, SGD
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, BatchNormalization, AveragePooling2D
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
#from sklearn.model_selection import train_test_split
#from tensorflow.keras.utils import plot_model
import matplotlib.image as mpimg
import os

######## VARIABLES ############

SAVE_MODEL = True

######## DATA LOADING #########
train = np.load("aux_data_pool\\train_data.npz")
test = np.load("aux_data_pool\\test_data.npz")
y_train_encoded = train['y']
y_test_encoded =  test['y']
x_train_normalized = train['x']
x_test_normalized = test['x']

######## CNN MODEL #########
pool_size = [(2,2),(5,5), (8,8)]
kernel_size = [(2,2),(5,5), (8,8)]
n_filt = [32]#[16, 32, 64]
#maxpooling + 2layer works well
for k in n_filt:
    print(k)
    print()
    print()
    model = Sequential()
    #model.add(Input(x_train_normalized.shape))
    model.add(Input(shape=(72, 48, 1)))
    model.add(Conv2D(32, kernel_size=(2,2), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(16, kernel_size=(5,5), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Flatten()) #input_shape=(32, 32, 2)
    #model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    #model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    #model.add(BatchNormalization())
    model.add(Dense(3, activation='softmax'))

    optimizer = Adam(learning_rate = 0.001)

    #optimizer = opt[k]
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.summary()

    #plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

    #if os.path.exists('model_architecture.png'):
    #     img = mpimg.imread('model_architecture.png')
    #     plt.figure
    #     plt.imshow(img)
    #     plt.axis('off')
    #     #plt.show(block=False)
    # else:
    #     print("Warning: The image 'model_architecture.png' was not created!")    

    early_stopping = EarlyStopping(monitor='val_loss', patience=7, verbose=1, restore_best_weights=True)

    ######## TRAINING #########

    #if LOAD_SAVED_MODEL:

    #else:                                                                                                 #0.5
    history = model.fit(x_train_normalized, y_train_encoded, epochs=30, batch_size=20, validation_data = (x_test_normalized, y_test_encoded), callbacks = early_stopping)#validation_split = 0.2   validation_split = 0.2, shuffle = True 

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
    
    #data = np.array( [history.history['loss'], history.history['val_loss'],  history.history['accuracy'], history.history['val_accuracy']])
    #np.save("history_pool\\history_nfilt_" + str(k)+".npy", data)

y_pred = model.predict(x_test_normalized)

y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test_encoded, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)
print(cm)
ConfusionMatrixDisplay.from_predictions(y_true_classes,y_pred_classes)

plt.show()