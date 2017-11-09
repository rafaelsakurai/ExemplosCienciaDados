from sklearn.linear_model import LinearRegression
import pandas as pd

df = pd.read_csv('../Dados/aptos.csv')
metros = df[['metros']].values
preco = df[['valor']].values

# Regressão Linear Simples

model = LinearRegression()
model.fit(metros, preco)

print ("Um apto de 100m custa: %.2f" % model.predict(100))