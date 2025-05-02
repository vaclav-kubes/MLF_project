
######## IMPORTS #########
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #hide info from CPU optimizer and Deep Neural Network Library 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv

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
#plot_model_once = True

# For saving history only when these match:
SAVE_HISTORY_FILTER = 32
SAVE_HISTORY_KERNEL = (5, 5)
SAVE_HISTORY_POOL = (2, 2)

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
best_params  = None   #(nf1, k1, p1, nf2, k2, p2, use_bn1, do1, use_bn2, do2, use_bn3)
best_history = None
best_val_loss = float('inf')
best_val_acc = 0
run_id = 0
results_file = "results.csv" #for saving the results through EVERY computed run

with open(results_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'run', 'nf1', 'k1', 'p1', 'nf2', 'k2', 'p2',
        'f1_score', 'val_loss', 'val_accuracy',
        'bn1', 'dropout1', 'bn2', 'dropout2', 'bn3'
    ])

######### HYPERPARAMETER GRID #########
pool_size = [(2,2), (3,3), (4,4), (5,5)]
n_filt = [16, 24, 32, 48, 64]
kernel_size = [(2,2), (5,5), (8,8)] 
add_layer = [0, 1]

######### DIMENSION CHECK FUNCTION #########
def is_valid_output_shape(input_shape, kernel_size, pool_size):
    h, w = input_shape
    h = (h - kernel_size[0]) + 1
    w = (w - kernel_size[1]) + 1
    if h <= 0 or w <= 0:
        return False
    h = (h - pool_size[0]) // pool_size[0] + 1
    w = (w - pool_size[1]) // pool_size[1] + 1
    return h > 0 and w > 0

######## CNN MODEL #########
for p1 in pool_size:
    for p2 in pool_size:
        if p1[0] > p2[0]: # pool size 1 always < pool size 2
            continue
        for nf1 in n_filt:
            nf2 = nf1 // 2
            for k1 in kernel_size:
                for k2 in kernel_size:
                    input_shape = (72, 48)
                    use_bn1 = int(np.random.choice(add_layer))
                    do1 = round(np.random.uniform(0.0, 0.5), 3)
                    use_bn2 = int(np.random.choice(add_layer))
                    do2 = round(np.random.uniform(0.0, 0.5), 3)
                    use_bn3 = int(np.random.choice(add_layer))
                    run_id += 1

                    ######## CONTROL OF THE SHAPE VALIDIY #########
                    if not is_valid_output_shape(input_shape, k1, p1):
                        continue
                    intermediate_shape = (
                        (input_shape[0] - k1[0]) + 1,
                        (input_shape[1] - k1[1]) + 1
                    )
                    intermediate_shape = (
                        (intermediate_shape[0] - p1[0]) // p1[0] + 1,
                        (intermediate_shape[1] - p1[1]) // p1[1] + 1
                    )

                    if not is_valid_output_shape(intermediate_shape, k2, p2):
                        continue

                    print(f"\n[RUN {run_id}] Training: Conv2D_1: f{nf1}, k1={k1}; Pool1={p1}; Conv2D_2: f{nf2}, k2={k2}; Pool2={p2}; bn1={use_bn1}, do1={do1}, bn2={use_bn2}, do2={do2}, bn3={use_bn3}")
                                        
                    ######## MODEL BUILDING #########
                    model = Sequential()
                    model.add(Input(shape=(72, 48, 1)))

                    model.add(Conv2D(nf1, kernel_size=k1, activation='relu'))
                    model.add(MaxPooling2D(pool_size=p1))

                    model.add(Conv2D(nf2, kernel_size=k2, activation='relu'))
                    model.add(MaxPooling2D(pool_size=p2))

                    model.add(Flatten()) 

                    if use_bn1:
                        model.add(BatchNormalization())

                    model.add(Dense(48, activation='relu'))

                    model.add(Dropout(do1))
                    
                    if use_bn2:
                        model.add(BatchNormalization())

                    model.add(Dense(25, activation='relu'))

                    model.add(Dropout(do2))

                    if use_bn3:
                        model.add(BatchNormalization())

                    model.add(Dense(3, activation='softmax')) # last layer for classication to bts1, bts2, bts3

                    optimizer = Adam(learning_rate = 0.001)
                    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                    #model.summary()
                    early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, restore_best_weights=True)

                    ######## TRAINING #########
                    try:
                        history = model.fit(
                            x_train_normalized, y_train_encoded,
                            epochs=30,
                            batch_size=20,
                            validation_split=0.2,
                            callbacks=early_stopping
                        )

                        y_pred = model.predict(x_test_normalized)
                        y_pred_classes = np.argmax(y_pred, axis=1)
                        y_true_classes = np.argmax(y_test_encoded, axis=1)

                        ######## SAVE HISTORY TO SEE THE INFLUENCE OF CHANGING NUMBER OF FILTERS #########
                        #if k1 == SAVE_HISTORY_KERNEL and k2 == SAVE_HISTORY_KERNEL and p1 == SAVE_HISTORY_POOL and p2 == SAVE_HISTORY_POOL:
                        #   data = np.array([
                        #      history.history['loss'],
                        #     history.history['val_loss'],
                            #    history.history['accuracy'],
                            #   history.history['val_accuracy']
                            #])
                            #np.save(f"history_pool\\history_nfilt_{nf1}.npy", data)
                            #print(f"[SAVED] History for {nf1} filters with fixed number of kernels: {SAVE_HISTORY_KERNEL}")

                        ######## SAVE HISTORY TO SEE THE INFLUENCE OF CHANGING KERNEL SIZE #########
                        #if nf1 == SAVE_HISTORY_FILTER and p1 == SAVE_HISTORY_POOL and p2 == SAVE_HISTORY_POOL:
                        #   data = np.array([
                        #      history.history['loss'],
                        #     history.history['val_loss'],
                            #    history.history['accuracy'],
                            #   history.history['val_accuracy']
                            #])
                        # np.save(f"history_pool\\history_kernel_size_{k1[0]}.npy", data)
                            #print(f"[SAVED] History for {k1} kernel size with fixed number of filters: {SAVE_HISTORY_FILTER}")

                        f1 = f1_score(y_true_classes, y_pred_classes, average='macro')
                        print(f"F1 score for [RUN {run_id}]: {f1:.5f}")

                        ######## UPDATE BEST MODEL #########
                        current_val_loss = history.history['val_loss'][-1]
                        current_val_acc  = history.history['val_accuracy'][-1]

                        with open(results_file, mode='a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                run_id, nf1, k1, p1, nf2, k2, p2,
                                round(f1, 5), round(current_val_loss, 5), round(current_val_acc, 5),
                                use_bn1, do1, use_bn2, do2, use_bn3
                            ])

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
                            best_config = (nf1, k1, p1, nf2, k2, p2, use_bn1, do1, use_bn2, do2, use_bn3)

                    except Exception as e:
                        print(f"[SKIPPED] Error in configuration: {nf1}, {k1}, {p1}, {nf2}, {k2}, {p2}, {use_bn1}, {do1}, {use_bn2}, {do2}, {use_bn3} → {e}")
                        continue

######## FINAL SAVE & DISPLAY #########
if best_model is not None:
    nf1, k1, p1, nf2, k2, p2, use_bn1, do1, use_bn2, do2, use_bn3 = best_config
    print(f"\nBest model: F1={best_f1:.4f} | CV1: {nf1}, {k1} | P1: {p1} | CV2: {nf2}, {k2} | P2: {p2} | BN1: {use_bn1} | D1: {do1} | BN2: {use_bn2} | D2: {do2} | BN3: {use_bn3} ")
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
