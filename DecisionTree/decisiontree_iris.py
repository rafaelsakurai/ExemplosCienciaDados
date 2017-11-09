import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
# Dataset https://archive.ics.uci.edu/ml/datasets/iris
nomes = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Class']
df = pd.read_csv('../Dados/iris.data', names = nomes)

caracteristicas = df.columns.difference(['Class'])
X = df[caracteristicas].values
y = df['Class'].values

# Decision Treen
iris_classificador = DecisionTreeClassifier(random_state=1234, criterion='entropy', max_depth=5)
iris_classificador.fit(X, y)

novos_exemplos = [[1.6,0.5,5.0,3.6], [4.2,1.2,5.8,2.7], [5.2,2.4,7.0,3.2]]
print (iris_classificador.predict(novos_exemplos))

scores_dt = cross_val_score(iris_classificador, X, y, scoring='accuracy', cv=5)
print (scores_dt.mean())

# Random Forest

from sklearn.ensemble import RandomForestClassifier

classifier_rf = RandomForestClassifier(random_state=1234, criterion='entropy', max_depth=5, n_estimators=5, n_jobs=-1)
scores_rf = cross_val_score(classifier_rf, X, y, scoring='accuracy', cv=5)

print (scores_rf.mean())