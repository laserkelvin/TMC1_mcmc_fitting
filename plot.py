
from matplotlib import pyplot as plt
import numpy as np

chains = np.load("prior/chain.npy", allow_pickle=True)

names = ["Tex", "VLSR4", "dV"]

for i, name in zip([-5, -2, -1], names):
    plt.plot(chains[:,:,i].T)
    plt.title(name)
    plt.show()
