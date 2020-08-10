
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

chains = np.load("prior/benzonitrile/chain.npy", allow_pickle=True)[:,-1000:,].reshape(-1, 14)
df = pd.DataFrame(chains, columns=["SS1", "SS2", "SS3", "SS4", "Ncol1", "Ncol2", "Ncol3", "Ncol4", "Tex", "vlsr1", "vlsr2", "vlsr3", "vlsr4", "dV"])

seed = np.random.seed(42)

g = sns.PairGrid(df, corner=True)
g.map_diag(plt.hist, edgecolor="k")
g.map_lower(sns.kdeplot)
g.fig.tight_layout()
plt.show()

#names = ["Tex", "VLSR4", "dV"]
#
#for i, name in zip([8, -2, -1], names):
#    plt.plot(chains[:,:,i].T)
#    plt.title(name)
#    plt.show()
