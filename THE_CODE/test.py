import numpy as np
import matplotlib.pyplot as plt 

data = np.array(0)
data = np.load("D:\\Users\\User\\Documents\\MLF_project\\MPA-MLF_data\\Train\\0.npy", "r")

print(data.shape )
plt.imshow(data)
plt.show()