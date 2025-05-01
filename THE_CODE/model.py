
######## IMPORTS #########
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #hide info from CPU optimizer and Deep Neural Network Library 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.models import Sequential
from keras.optimizers import Adam, AdamW, SGD
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, BatchNormalization, Dropout, AveragePooling2D
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
best_val_loss = float('inf')
best_val_acc = 0

######### HYPERPARAMETER GRID #########
n_filt = [16, 32, 64]
kernel_size = [(2,2), (5,5), (8,8)]

######## CNN MODEL #########
for nf1 in n_filt:
    nf2 = nf1 // 2
    for k1 in kernel_size:
        for k2 in kernel_size:
            print(f"\nTraining: Conv2D_1: filters={nf1}, kernel={k1}; Conv2D_2: filters={nf2}, kernel={k2}...")
            
            ######## MODEL BUILDING #########
            model = Sequential()
            #model.add(Input(x_train_normalized.shape))
            model.add(Input(shape=(72, 48, 1)))
            model.add(Conv2D(nf1, kernel_size=k1, activation='relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(Conv2D(nf2, kernel_size=k2, activation='relu'))
            model.add(MaxPooling2D(pool_size=(3,3)))
            model.add(Flatten()) 
            model.add(BatchNormalization())
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.3))
            model.add(BatchNormalization())
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.3))
            model.add(BatchNormalization())
            model.add(Dense(3, activation='softmax'))

            optimizer = Adam(learning_rate = 0.001)

            #optimizer = opt[k]
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

            #model.summary()

            early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)

            ######## TRAINING #########
            history = model.fit(
                x_train_normalized, y_train_encoded,
                epochs=30,
                batch_size=15,
                validation_split=0.3,
                callbacks=early_stopping
            ) # validation_data=(x_test_normalized, y_test_encoded),shuffle = True, epochs=30, batch_size=20

            y_pred = model.predict(x_test_normalized)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_test_encoded, axis=1)

            ######## SAVE HISTORY TO SEE THE INFLUENCE OF CHANGING NUMBER OF FILTERS #########
            if k1 == SAVE_HISTORY_KERNEL and k2 == SAVE_HISTORY_KERNEL:
                data = np.array([
                    history.history['loss'],
                    history.history['val_loss'],
                    history.history['accuracy'],
                    history.history['val_accuracy']
                ])
                np.save(f"history_pool\\history_nfilt_{nf1}.npy", data)
                print(f"[SAVED] History for {nf1} filters with fixed number of kernels: {SAVE_HISTORY_KERNEL}")

            ######## SAVE HISTORY TO SEE THE INFLUENCE OF CHANGING KERNEL SIZE #########
            if nf1 == SAVE_HISTORY_FILTER:
                data = np.array([
                    history.history['loss'],
                    history.history['val_loss'],
                    history.history['accuracy'],
                    history.history['val_accuracy']
                ])
                np.save(f"history_pool\\history_kernel_size_{k1[0]}.npy", data)
                print(f"[SAVED] History for {k1} kernel size with fixed number of filters: {SAVE_HISTORY_FILTER}")

            f1 = f1_score(y_true_classes, y_pred_classes, average='macro')
            print(f"F1 score for Conv2D_1: filters={nf1}, kernel={k1}; Conv2D_2: filters={nf2}, kernel={k2}: {f1:.4f}")

            ######## UPDATE BEST MODEL #########
            current_val_loss = history.history['val_loss'][-1]
            current_val_acc  = history.history['val_accuracy'][-1]

            if (
                f1 > best_f1 or
                (f1 == best_f1 and current_val_loss < best_val_loss) or
                (f1 == best_f1 and current_val_loss == best_val_loss and current_val_acc > best_val_acc)
            ):
                best_f1 = f1
                best_val_loss = current_val_loss
                best_val_acc = current_val_acc
                best_model = model
                best_pred = y_pred_classes
                best_true = y_true_classes
                best_history = history.history
                best_config = (nf1, k1, nf2, k2)


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

    save_model_arch = "model_architecture.png"
    plot_model(best_model, to_file=save_model_arch, show_shapes=True, show_layer_names=True)

    if os.path.exists(save_model_arch):
        img = mpimg.imread(save_model_arch)
        plt.figure()
        plt.imshow(img)
        plt.axis('off')
        plt.show(block=False)

    cm = confusion_matrix(best_true, best_pred)
    print("Confusion Matrix:")
    print(cm)
    ConfusionMatrixDisplay.from_predictions(best_true, best_pred)
    plt.show()