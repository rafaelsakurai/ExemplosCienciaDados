import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score

# Dataset https://archive.ics.uci.edu/ml/datasets/iris
nomes = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Class']
df = pd.read_csv('../Dados/iris.data', names = nomes)

caracteristicas = df.columns.difference(['Class'])
X = df[caracteristicas].values
y = df['Class'].apply(lambda x : {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}[x]).values

# K-Means

iris_kmeans = KMeans(n_clusters=3)
iris_kmeans.fit(X)

novos_exemplos = [[1.6,0.5,5.0,3.6], [4.2,1.2,5.8,2.7], [5.2,2.4,7.0,3.2]]
print (iris_kmeans.predict(novos_exemplos))

# Cross Validation

scores_dt = cross_val_score(iris_kmeans, X, y, scoring='accuracy', cv=5)
print (scores_dt.mean())