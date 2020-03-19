import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from adaline import Adaline

df = pd.read_csv('Numbers.csv', header=None)
df.head()

x = df.iloc[0:21,0].values
y = df.iloc[0:21,1].values

cm_bright = ListedColormap(['#0000FF', '#FF0000'])
plt.scatter([r[0] for index, r in df.iterrows()], [r[0] for index, r in df.iterrows()], c=y, cmap=cm_bright)
plt.scatter(None, None, color='r', label='Ímpares')
plt.scatter(None, None, color='b', label='Pares')
plt.legend()
plt.title('Visualizacao dos números (Ímpares e Pares)')
plt.savefig('train.png')

adaline = Adaline(1)
adaline.train(x, y)

## Test 1
A = 7 # Impares (1)
predict = adaline.predict(A)
print('## Teste 1')
print('Entrada: ', A)
print('Número: Ímpar')
if predict == 1:
  print('Previsão: Ímpar')
else:
  print('Previsão: Par')

print('')

## Test 2
B = 12 # Par (-1)
predict = adaline.predict(B)
print('## Teste 2')
print('Entrada: ', B)
print('Número: Par')
if predict == 1:
  print('Previsão: Ímpar')
else:
  print('Previsão: Par')