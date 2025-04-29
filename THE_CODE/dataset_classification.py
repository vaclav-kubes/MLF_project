import numpy as np
import extract_data as ed
import matplotlib.pyplot as plt
from keras import saving

train_dataset = ed.load_data("MPA-MLF_data\\Train", (72, 48))
unknown_data = ed.load_data("MPA-MLF_data\\Test", (72, 48))
mean_dev = np.load("THE_CODE\\mean_dev.npy")
#mean = np.mean(train_dataset) #Z-score normalization of training data
#deviation = np.std(train_dataset)
print(mean_dev[0], mean_dev[1])
unknown_data_norm = (unknown_data - mean_dev[0]) / mean_dev[1]

model = saving.load_model("THE_CODE\\model.keras")

pred = model.predict(unknown_data_norm)#.reshape(-1, 72, 48, 1)
array_to_save = np.vstack((np.arange(0, 119 + 1), np.argmax(pred, axis=1))).astype(np.int16)
np.savetxt("THE_CODE\\my_guess.csv", array_to_save.transpose(),"%d", ",", header="ID,target", comments="")

