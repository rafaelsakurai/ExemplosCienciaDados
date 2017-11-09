import pandas as pd
# Dataset https://archive.ics.uci.edu/ml/datasets/iris
nomes = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Class']
df = pd.read_csv('../Dados/iris.data', names = nomes)

X = df[df.columns.difference(['Class'])].values
y = df['Class'].values

### SVM

from sklearn.svm import LinearSVC

iris_classificador = LinearSVC()
iris_classificador.fit(X, y)

novos_exemplos = [[1.6,0.5,5.0,3.6], [4.2,1.2,5.8,2.7], [5.2,2.4,7.0,3.2]]
print iris_classificador.predict(novos_exemplos)

### Cross Validation

from sklearn.model_selection import cross_val_score

scores_dt = cross_val_score(iris_classificador, X, y, scoring='accuracy', cv=5)
print scores_dt.mean()