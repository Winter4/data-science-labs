import pandas
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering

db = pandas.read_csv('data.csv')
print(db)

X = db.loc[:, 'variance':'entropy']
Y = db['class']

model = SpectralClustering(n_clusters = 2, assign_labels = 'discretize', affinity = 'rbf', n_neighbors = 17)
model.fit(X, Y)

predict = model.fit_predict(X)

counter = 0
for i in range(len(predict)):
  if (predict[i] == Y[i]):
    counter += 1

print()
print("Точность предсказания: {:d} * 100 / {:d} = {:.2f}%".format(counter, len(predict), counter * 100 / len(predict)))

plt.scatter(db['variance'], db['kurtosis'], c = predict)
plt.show()