import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import b2plot


plt.style.use('belle2')



import numpy as np
import pandas as pd
df = {'mass': np.append(np.random.random_sample(1000)*7 - 3.5, np.random.normal(0, 0.5, 1000))}
df = pd.DataFrame(df)


b2plot.errorbar(df.sample(1500).mass, color='black', label='Data')
b2plot.hist(df.sample(1500).mass, fill=True, lw=2, style=1, label='MC10')
plt.legend()
b2plot.watermark(fontsize=20, px=.2)
b2plot.xlim()
b2plot.labels('E', "Events", "GeV", 1)

plt.savefig("example_hist_errorbar.pdf")