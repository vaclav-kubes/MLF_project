import matplotlib.pyplot as plt
import numpy as np

nfilt16 = np.load("..\\history_pool\\history_nfilt_16.npy")
nfilt32 = np.load("..\\history_pool\\history_nfilt_32.npy")
nfilt64 = np.load("..\\history_pool\\history_nfilt_64.npy")
#x_axis = range(1, len(adam[1])+1)

print(np.min(nfilt16[2]), np.min(nfilt16[3]))
print(np.min(nfilt32[2]), np.min(nfilt32[3]))
print(np.min(nfilt64[2]), np.min(nfilt64[3]))


plt.figure()
plt.semilogy(range(1, len(nfilt16[1])+1),np.abs( nfilt16[0]),"r--", range(1, len(nfilt16[1])+1), np.abs(nfilt16[1]), "r-.")
plt.semilogy( range(1, len(nfilt32[1])+1), np.abs(nfilt32[0]),"b--", range(1, len(nfilt32[1])+1), np.abs(nfilt32[1]), "b-.")
plt.semilogy(range(1, len(nfilt64[1])+1), np.abs(nfilt64[0]),"g--", range(1, len(nfilt64[1])+1), np.abs(nfilt64[1]), "g-.")
plt.grid(True, 'both', 'both')
plt.xlabel("Epoch")
plt.title("Loss")
plt.minorticks_on()
plt.xticks(range(1,30,2))
plt.xlim(left=1, right=30)
plt.legend(['Trainig: Conv. filters 16', "Validation: Conv. filters 16", 'Trainig: Conv. filters 32', "Validation: Conv. filters 32", 'Trainig: Conv. filters 64', "Validation: Conv. filters 64"])


plt.figure()
#fig, ax1 = plt.subplots()
#ax2 = ax1.twinx()
plt.plot(range(1, len(nfilt16[1])+1), nfilt16[2],"r--", range(1, len(nfilt16[1])+1), nfilt16[3], "r-.")
#plt.yscale('log')

plt.plot( range(1, len(nfilt32[1])+1), nfilt32[2],"b--", range(1, len(nfilt32[1])+1), nfilt32[3], "b-.")
#plt.yscale('log')

plt.plot(range(1, len(nfilt64[1])+1), nfilt64[2],"g--", range(1, len(nfilt64[1])+1), nfilt64[3], "g-.")
#plt.yscale('log')

plt.grid(True, 'both', 'both')
plt.xlabel("Epoch")
plt.title("Accuracy")
plt.minorticks_on()
plt.xticks(range(1,30,2))
plt.xlim(left=1, right=30)
plt.legend(['Trainig: Conv. filters 16', "Validation: Conv. filters 16", 'Trainig: Conv. filters 32', "Validation: Conv. filters 32", 'Trainig: Conv. filters 64', "Validation: Conv. filters 64"])
plt.show()

"""
fig, ax = plt.subplots()
#ax2=ax.twiny()
ax.plot(range(1, len(adam[1])+1), adam[2],"r--", label="Training: Adam")
ax.plot(range(1, len(adam[1])+1), adam[3], "r-.", label="Validation: Adam")
ax.plot(range(1, len(adamw[1])+1), adamw[2], "b--", label="Training: AdamW")
ax.plot(range(1, len(adamw[1])+1), adamw[3], "b-.", label="Validation: AdamW")
ax.plot(range(1, len(sgd[1])+1), sgd[2], "g--", label="Training: SGD")
ax.plot(range(1, len(sgd[1])+1), sgd[3], "g-.", label="Validation: SGD")

ax.set_yscale('log')  # extra assurance
#ax2.set_yscale('log')
ax.grid(True, which='both')
ax.set_xlabel("Epoch")
ax.set_title("Accuracy")
ax.set_xticks(range(1, 28, 2))
ax.set_xlim(left=1, right=27)
#ax.set_ylim( top=1.001)
ax.legend()
plt.show()
"""
"""
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()

# Plot Adam and SGD on the primary y-axis
ax1.plot(range(1, len(adam[1]) + 1), adam[2], "r--", label="Training: Adam")
ax1.plot(range(1, len(adam[1]) + 1), adam[3], "r-.", label="Validation: Adam")
ax1.plot(range(1, len(sgd[1]) + 1), sgd[2], "g--", label="Training: SGD")
ax1.plot(range(1, len(sgd[1]) + 1), sgd[3], "g-.", label="Validation: SGD")

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy (Adam, SGD)")
#ax1.set_xticks(range(1, 31, 2).append(30))
ax1.set_xlim(left=1, right=30)
#ax1.set_yscale('log')
#ax1.set_yticks(np.logspace(0.084, 1, 9, True))
ax1.grid(True, which='both')
ax1.set_title("Accuracy with Second Y Axis for AdamW")

# Create a secondary y-axis for AdamW
ax2 = ax1.twinx()
ax2.plot(range(1, len(adamw[1]) + 1), adamw[2], "b--", label="Training: AdamW")
ax2.plot(range(1, len(adamw[1]) + 1), adamw[3], "b-.", label="Validation: AdamW")
ax2.set_ylabel("Accuracy (AdamW)")
##ax2.set_yscale('log')

# Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower right")

plt.show()
"""
