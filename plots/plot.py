# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import hyperactive
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# import matplotlib as mpl
import seaborn as sns

ml = MultipleLocator(5)

version = str("_v" + hyperactive.__version__)
n_iter = 100

sns.set(style="whitegrid")
data = pd.read_csv("./data/optimizer_calc_time_v0.3.3", header=0)
# print("data", data)

columns = data.columns
values = data.values
# print("values", values, values.shape)
# print("values[:, 0]", values[:, 0])

no_opt = values[:, 0]

values_norm = values / no_opt[:, None]

# print("values_norm", values_norm, values_norm.shape)

fig, ax = plt.subplots()
plt.figure(figsize=(15, 5))
# plt.grid(True)
# plt.ylabel(r"$\dfrac{T_{norm}}{iteration}$", rotation=0)
plt.ylabel(r"$T_{norm}$", rotation=0)

data = pd.DataFrame(values_norm, columns=columns)
ax = sns.barplot(data=data, alpha=0.75, capsize=0.1)
# ax = sns.barplot(x=columns, y=values_norm, alpha=0.75, ci="sd")
ax.set_xticklabels(ax.get_xticklabels(), rotation=65)

# ax.tick_params(axis="x", which="minor", bottom=False)
# ax.yaxis.set_minor_locator(ml)

y_min = 0.99
y_max = 1.02
step_major = 0.01
step_minor = 0.025

ax.set_ylim(y_min, y_max)

ticks_major = list(np.arange(y_min, y_max, step_major))
ticks_minor = list(np.arange(y_min, y_max, step_minor))

ax.yaxis.set_ticks(ticks_major)
ax.yaxis.set_ticks(ticks_minor, minor=True)

ax.xaxis.set_label_coords(-0.05, 0.5)
ax.get_yaxis().set_label_coords(-0.05, 1.1)

# ax.yaxis.grid(which="minor", color="r", linestyle="-", linewidth=2)

fig = ax.get_figure()
fig.tight_layout()
fig.savefig("optimizer_time.png")
