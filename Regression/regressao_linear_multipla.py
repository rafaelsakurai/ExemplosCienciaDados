from sklearn.linear_model import LinearRegression
import pandas as pd

df = pd.read_csv('../Dados/aptos.csv')
X = pd.get_dummies(df[['metros', 'quartos', 'vagas', 'reformado', 'bairro']]).values
y = df[['valor']].values

# Regressão Linear Múltipla

model = LinearRegression()
model.fit(X, y)

print ("Um apto de 100m com 3 quartos, 2 vagas, reformado no Centro custa: %.2f" % model.predict([[100, 3, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
print ("Um apto de 100m com 3 quartos, 2 vagas, reformado no Rudge custa: %.2f" % model.predict([[100, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,]]))