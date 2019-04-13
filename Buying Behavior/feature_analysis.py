import numpy as np
import pandas as pd
import seaborn as sn
import missingno as msno
import matplotlib.pyplot as plt

# Read data
data = pd.read_csv("wholesale_customers.csv")
# Exclude Region and Channel
data.drop(['Region', 'Channel'], axis = 1, inplace = True)

# Display the data
print(data.describe())

# Missing data detection
msno.matrix(data,figsize=(10,3))

# Data distribution
fig, axes = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(10, 7)
sn.boxplot(data=data, orient="v")

# Correlation analasys
corrMatt = data.corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sn.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)

# Scatterplot
mx_plot = sn.pairplot(data, diag_kind="kde", size=1.6)
mx_plot.set(xticklabels=[])