from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd

# Dataset https://archive.ics.uci.edu/ml/datasets/sms+spam+collection

df = pd.read_csv('../Dados/SMSSpamCollection', delimiter='\t', header=None)

# Separa o dataset em treino (75%) e teste (25%)

x_treino_texto, x_teste_texto, y_treino, y_teste = train_test_split(df[1], df[0])

# Aplica TF-IDF nos textos

vectorizer = TfidfVectorizer()
vectorizer.fit(x_treino_texto)
x_treino = vectorizer.transform(x_treino_texto)
x_teste = vectorizer.transform(x_teste_texto)

# Regressão Logistica

model = LogisticRegression()
model.fit(x_treino, y_treino)

# Faz a predição e apresenta 100 exemplos

preditos = model.predict(x_teste)
for i in range(0,100):
  print "A mensagem \"{}\" foi considerada {} e eh {}".format(x_teste_texto.values[i], preditos[i], y_teste.values[i])

# Cross Validation
scores_dt = cross_val_score(model, vectorizer.fit_transform(df[1]), df[0], scoring='accuracy', cv=5)
print scores_dt.mean()