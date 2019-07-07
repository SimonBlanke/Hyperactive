# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import hyperactive
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import matplotlib as mpl
import seaborn as sns


version = str("_v" + hyperactive.__version__)
n_iter = 300

sns.set(style="whitegrid")
data = pd.read_csv("./data/optimizer_calc_time_v0.3.3", header=0)

columns = data.columns
values = data.values[0]
values_1 = values / 100
values_norm = values_1 / values_1[0]

fig, ax = plt.subplots()
plt.figure(figsize=(12, 4))
plt.grid(True)
plt.ylabel(r"$\dfrac{T_{norm}}{iteration}$", rotation=0)

ax = sns.barplot(x=columns, y=values_norm, alpha=0.75)
ax.set_xticklabels(ax.get_xticklabels(), rotation=55)

y_min = 0.99
y_max = 1.02
ax.set_ylim(y_min, y_max)

ticks = list(np.arange(y_min, y_max, 0.01))

ax.yaxis.set_ticks(ticks)
ax.get_yaxis().set_label_coords(-0.05, 1.1)

fig = ax.get_figure()
fig.tight_layout()
fig.savefig("optimizer_time.png")
