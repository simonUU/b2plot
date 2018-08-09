
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import b2plot


ns =3000
nb =8000

plt.style.use("belle2")

df = {'mass': np.append(np.random.random_sample(nb)*7 - 3.5, np.random.normal(0, 0.5, ns))}
df['sig'] = np.append(np.zeros(nb), np.ones(ns), )
df['exp'] = np.random.randint(0, 4, ns+nb)
df = pd.DataFrame(df)

b2plot.stacked(df, "mass", 'exp', bins=50, color=b2plot.b2helix(4), label=range(4))
b2plot.errorbar(df.mass.values, color='black', weights=np.random.normal(1.01,0.9, len(df)), label="Data")
b2plot.xlim()
plt.legend()
b2plot.labels("$M$", "Events", 'GeV',)

plt.savefig("example_stacked.pdf")