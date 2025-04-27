import numpy as np
import matplotlib.pyplot as plt 

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 10,
        }

data = np.array(0)
data = np.load("D:\\Users\\User\\Documents\\MLF_project\\MPA-MLF_data\\Train\\0.npy", "r")

print(data.shape )
ax = plt.imshow(data, cmap='gray')
plt.xlabel("Number of reperitions [-]", font)
plt.ylabel("Subcarrier [-]", font)
ax.axes.invert_yaxis()
plt.title("Channel freq. response determined from PSS and SSS", font)
plt.show()