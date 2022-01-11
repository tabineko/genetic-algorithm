import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('hist_crossover.csv', header=None)

# print(df.values.tolist())
plt.boxplot(df.values.tolist(), labels=list(map(str, [0.9, 0.7, 0.5, 0.3, 0.1])), showfliers=False)
plt.ylabel('generation')
plt.xlabel('crossover probability')
plt.show()


