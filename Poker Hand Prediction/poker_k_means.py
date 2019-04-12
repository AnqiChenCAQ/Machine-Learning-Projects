import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA 

# Data Preparation
features = ['Card 1 Suit', 'Card 1 Rank', 'Card 2 Suit', 'Card 2 Rank', 'Card 3 Suit', 'Card 3 Rank', 'Card 4 Suit', 'Card 4 Rank', 'Card 5 Suit', 'Card 5 Rank', 'Poker Hand']
train_data = pd.read_csv('poker_hand_train.csv', sep = ',', names = features)
test_data = pd.read_csv('poker_hand_test.csv', sep = ',', names = features)

data = pd.concat([train_data, test_data])

categoryVariableList  = ['Card 1 Suit', 'Card 1 Rank', 'Card 2 Suit', 'Card 2 Rank', 'Card 3 Suit', 'Card 3 Rank', 'Card 4 Suit', 'Card 4 Rank', 'Card 5 Suit', 'Card 5 Rank', 'Poker Hand']
for var in categoryVariableList:
    data[var] = data[var].astype('float64')
    
# loop through clusters and fit the model to the train set
clusters=range(1,11)
meandist=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(data)
    clusassign=model.predict(data)
    meandist.append(sum(np.min(cdist(data, model.cluster_centers_, 'euclidean'), axis=1))
    / data.shape[0])
    
# Display the data
plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method') # pick the fewest number of clusters that reduces the average distance


# Interpret 2 cluster solution
model3=KMeans(n_clusters=2)
model3.fit(data)
clusassign=model3.predict(data)

# Squash the data into 2D
pca_2 = PCA(2) # Two Canonical Variables
plot_columns = pca_2.fit_transform(data)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model3.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 2 Clusters')
plt.show()