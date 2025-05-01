
######## IMPORTS #########
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #hide info from CPU optimizer and Deep Neural Network Library 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.models import Sequential
from keras.optimizers import Adam, AdamW, SGD
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, BatchNormalization, AveragePooling2D
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
#from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model

######### FIXED SETTINGS #########
SAVE_MODEL = True
plot_model_once = True

# For saving history only when these match:
SAVE_HISTORY_FILTER = 32
SAVE_HISTORY_KERNEL = (5, 5)

######## DATA LOADING #########
train = np.load("aux_data_pool\\train_data.npz")
test = np.load("aux_data_pool\\test_data.npz")
y_train_encoded = train['y']
y_test_encoded =  test['y']
x_train_normalized = train['x']
x_test_normalized = test['x']

######### TRACK BEST MODEL #########
best_f1      = 0
best_model   = None
best_preds   = None
best_true    = None
best_params  = None   # (nf1, k1, nf2, k2)
best_history = None

######### HYPERPARAMETER GRID #########
n_filt = [32, 64]
kernel_size = [(2,2), (5,5)]

######## CNN MODEL #########
for nf1 in n_filt:
    nf2 = nf1 // 2

    for k1 in kernel_size:
        for k2 in kernel_size:
            print(f"\nTraining: Conv2D_1: filters={nf1}, kernel={k1}; Conv2D_2: filters={nf2}, kernel={k2}")
            
            ######## MODEL BUILDING #########
            model = Sequential()
            #model.add(Input(x_train_normalized.shape))
            model.add(Input(shape=(72, 48, 1)))
            model.add(Conv2D(nf1, kernel_size=k1, activation='relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(Conv2D(nf2, kernel_size=k2, activation='relu'))
            model.add(MaxPooling2D(pool_size=(3,3)))
            model.add(Flatten()) 
            #model.add(BatchNormalization())
            model.add(Dense(64, activation='relu'))
            #model.add(BatchNormalization())
            model.add(Dense(32, activation='relu'))
            #model.add(BatchNormalization())
            model.add(Dense(3, activation='softmax'))

            optimizer = Adam(learning_rate = 0.1) #learning_rate = 0.001)

            #optimizer = opt[k]
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

            #model.summary()
            """
            if plot_model_once:
                save_model_arch = f"model_arch_k{nf1}_k{nf2}_{k1[0]}x{k1[1]}_{k2[0]}x{k2[1]}.png"
                plot_model(model, to_file=save_model_arch, show_shapes=True, show_layer_names=True)

                if os.path.exists(save_model_arch):
                    img = mpimg.imread(save_model_arch)
                    plt.figure()
                    plt.imshow(img)
                    plt.axis('off')
                    plt.show(block=False)
                plot_model_once = False 
            """
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True) #patience=7

            ######## TRAINING #########

            #if LOAD_SAVED_MODEL:

            #else:                                                                                                 #0.5
            history = model.fit(
                x_train_normalized, y_train_encoded,
                epochs=3,
                batch_size=2000,
                validation_split=(x_test_normalized, y_test_encoded),
                callbacks=[early_stopping]
            ) #validation_split = 0.2, shuffle = True, epochs=30, batch_size=20

            y_pred = model.predict(x_test_normalized)

            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_test_encoded, axis=1)

            f1 = f1_score(y_true_classes, y_pred_classes, average='macro')
            print(f"F1 score for Conv2D_1: filters={nf1}, kernel={k1}; Conv2D_2: filters={nf2}, kernel={k2}: {f1:.4f}")

            ######## UPDATE BEST MODEL #########
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_pred = y_pred_classes
                best_true = y_true_classes
                best_history = history.history
                best_config = (nf1, k1, nf2, k2)

            ######## SAVE HISTORY TO SEE INFLUENCE OF CHANGING FILTER #########
            if kernel_size == SAVE_HISTORY_KERNEL:
                data = np.array([
                    history.history['loss'],
                    history.history['val_loss'],
                    history.history['accuracy'],
                    history.history['val_accuracy']
                ])
                np.save(f"history_pool\\history_nfilt_{nf1}.npy", data)
                print(f"[SAVED] History for {nf1} filters with fixed number of kernels: {SAVE_HISTORY_KERNEL}")

            ######## SAVE HISTORY TO SEE INFLUENCE OF CHANGING KERNEL #########
            if n_filt == SAVE_HISTORY_FILTER:
                data = np.array([
                    history.history['loss'],
                    history.history['val_loss'],
                    history.history['accuracy'],
                    history.history['val_accuracy']
                ])
                np.save(f"history_pool\\history_kernel_size_{k1[0]}.npy", data)
                print(f"[SAVED] History for {kernel_size} kernel size with fixed number of filters: {SAVE_HISTORY_FILTER}")


            """
            data = np.array([
                history.history['loss'],
                history.history['val_loss'],
                history.history['accuracy'],
                history.history['val_accuracy']
            ])
            
            np.save(f"history_pool\\history_nfilt_{k}.npy", data)

            #if SAVE_MODEL: model.save("THE_CODE\\model.keras")

            ######## EVALUATION #########

            score = model.evaluate(x_test_normalized, y_test_encoded, verbose=1)
            val_loss = score[0]
            val_accuracy = score[1]
            print('Test loss:', score[0])
            print(f'Test accuracy: {score[1]*100} %')
            
            if val_accuracy > best_accuracy or (val_accuracy == best_accuracy and val_loss < best_loss):
                    best_accuracy = val_accuracy
                    best_loss = val_loss
                    best_y_pred = model.predict(x_test_normalized)
                    best_y_true = y_test_encoded
                    best_k = k"""

            """
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
            
            plt.show(block=False)
            
            
            data = np.array( [history.history['loss'], history.history['val_loss'],  history.history['accuracy'], history.history['val_accuracy']])
            np.save("history_pool\\history_nfilt_" + str(k)+".npy", data)
            """


######## FINAL SAVE & DISPLAY #########
if best_model is not None:
    nf1, ks1, nf2, ks2 = best_config
    print(f"\nBest model: F1={best_f1:.4f} | L1: {nf1}, {ks1} | L2: {nf2}, {ks2}")
    best_model.save("THE_CODE\\model.keras")

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title("Loss (Best Model)")
    plt.plot(best_history['loss'], label='Train')
    plt.plot(best_history['val_loss'], label='Validation')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.title("Accuracy (Best Model)")
    plt.plot(best_history['accuracy'], label='Train')
    plt.plot(best_history['val_accuracy'], label='Validation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

    cm = confusion_matrix(best_true, best_pred)
    print("Confusion Matrix:")
    print(cm)
    ConfusionMatrixDisplay.from_predictions(best_true, best_pred)
    plt.show()

"""
y_pred = model.predict(x_test_normalized)

y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test_encoded, axis=1)


cm = confusion_matrix(y_true_classes, y_pred_classes)
print(cm)
ConfusionMatrixDisplay.from_predictions(y_true_classes,y_pred_classes)

plt.show()
"""