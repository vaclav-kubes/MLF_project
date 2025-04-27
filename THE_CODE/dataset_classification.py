import numpy as np
import extract_data as ed
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras import saving

train_dataset = ed.load_data("MPA-MLF_data\Train", (72, 48))
unknown_data = ed.load_data("MPA-MLF_data\Test", (72, 48))
mean = np.mean(train_dataset) #Z-score normalization of training data
deviation = np.std(train_dataset)
unknown_data_norm = (unknown_data - mean) / deviation

model = saving.load_model("THE_CODE\\model.keras")

pred = model.predict(unknown_data_norm)
array_to_save = np.vstack((np.arange(0, 119 + 1), np.argmax(pred, axis=1))).astype(np.int16)
np.savetxt("THE_CODE\\my_guess.csv", array_to_save.transpose(),"%d", ",", header="ID,target", comments="")

