import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from readit import read_diamonds

df = read_diamonds()
fig, axs = plt.subplots(1, 2, figsize=(20, 8))
sns.lineplot(data=df, x='carat', y='price', ax=axs[0])
sns.scatterplot(data=df, x='carat', y='price', ax=axs[1])

axs[0].set_title('Lineplot for carat vs price')
axs[1].set_title('Scatterplot for carat vs price')

plt.savefig('figs/carat_vs_price.png')
plt.show()
