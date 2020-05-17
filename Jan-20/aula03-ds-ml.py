# Exemplo carros nas concessionárias
# Modelo de ML de árvore de decisão
# Algorítimo de classificação

import pandas as pd
df = pd.read_excel('~/aphonso/Ap/DSBD_Git/Data Science do zero - Aula ML/carros_usados.xls')

df.head()

df.describe()

# Identificar missing values no dataframe:
df.isnull().sum()

import seaborn as sns
from matplotlib import pyplot as plt
plt.style.use('ggplot')

plt.figure(1)

# Histogramas:
sns.distplot(df['total.cost'], color='green', kde=False)
plt.title('Distribuição da coluna total.cost')
# Podemos observar que a maioria dos carros custam em torno de 3k e 6k.

# Qual a quantidade média de dias que os carros ficam na concessionária?
plt.figure(2)
sns.distplot(df['lot.sale.days'],color='red', kde=False)
plt.title('Distribuição da coluna lot.sale.days')
# Podemos observar que os dias de vendas são inclinados para o lado esquerdo
# isso nos mostra que a maioria dos são vendidos nos primeiros 90 dias.
# Os restantes 20% são vendidos após 20 dias e são vendidos a preço de desconto.

# Qual a Kilometragem média dos veículos da concessionária?
plt.figure(3)
sns.distplot(df['mileage'],color='blue', kde=False)
plt.title('Distribuição da coluna mileage')
# Podemos observar que a grande maioria dos veículos estão na faixa de
# 60.000 a 90.000 de kilometragem

plt.show()

# Correlação:
df[['mileage','vehicle.age','lot.sale.days','total.cost']].head()
df[['mileage','vehicle.age','lot.sale.days','total.cost']].corr()
correlacoes = df[['mileage','vehicle.age','lot.sale.days','total.cost']].corr()

# Plot o mapa de calor para visualizar as correlações
# Paleta de cores: "YlGnBu"
sns.heatmap(correlacoes, annot=True, cmap="YlGnBu")
plt.show()

# Conclusões:
# O gráfico de calor acima nos mostra que não existe uma correlação forte das variáveis com a quantidade de dias de vendas(lot.sale.days)
# Podemos ver que a kilometragem tem uma correlação positiva com a idade do veículo.
# Podemos ver também que os dias de vendas (lot.sale.days) tem uma correlação positiva com o custo total do veículo (total.cost) e com a idade do veículo (vehicle.age)


# Dispersão entre os dias de vendas e outras variáveis:
# Caixa: amplitude
# Boxplot faz o agrupamento por quartis:
  # Linha do meio: mediana (2º quartil)
  # Linha de baixo: 1º quartil
  # Linha de cima: 3º quartil
# Pontos fora da linha de cima: considerados outliers

sns.scatterplot(x='lot.sale.days', y="total.cost", data=df)

sns.scatterplot(x='lot.sale.days', y="mileage", color='green', data=df)


# Tempo médio de vendas por tipo de veículos:
sns.boxplot(x="domestic.import", y="lot.sale.days", data=df)

# Tempo médio de vendas por categoria de veículos:
sns.boxplot(x="vehicle.type", y="lot.sale.days", data=df)

# Tempo médio de vendas por cor de veículos:
sns.boxplot(x="color.set", y="lot.sale.days", data=df)

# Tempo médio de vendas por marca dos veículos:
sns.boxplot(x="makex", y="lot.sale.days", data=df)
plt.xticks(size=5) # Tamanho da fonte dos labels do eixo X

# Tempo médio de vendas por estado:
sns.boxplot(x="state", y="lot.sale.days", data=df)


# Modelo de ML:
# Classificar se um carro vai passar de 90 dias na concessionária ou não. - Algorítimo de classificação.

# Pre-processando os dados:
# - Transformar colunas categóricas em numéricas, porque o algorítimo não processa dados categóricos.
# - Precisamos pré-processar algumas colunas, pois, são colunas categóricas.
# - Nesta etapa também removemos colunas não importantes para o modelo.

# LabelEncoder():
  # Para cada valor único da categoria, a coluna recebe um valor inteiro.

# Apagando coluna no pandas: colunas que não precisamos para o modelo
# O axis=1 indica que é uma coluna.
df.drop('vehicle.age.group', axis=1, inplace=True)
df.drop('data.set', axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder

# Instanciado os objetos LabelEncoder:
label_encoder1 = LabelEncoder()
label_encoder2 = LabelEncoder()
label_encoder3 = LabelEncoder()
label_encoder4 = LabelEncoder()
label_encoder5 = LabelEncoder()
label_encoder6 = LabelEncoder()

# Aplicar o LabelEncoder nos dados categóricos:
df['vehicle.type'] = label_encoder1.fit_transform(df['vehicle.type'])
df['domestic.import'] = label_encoder2.fit_transform(df['domestic.import'])
df['color.set'] = label_encoder3.fit_transform(df['color.set'])
df['makex'] = label_encoder4.fit_transform(df['makex'])
df['state'] = label_encoder5.fit_transform(df['state'])
df['make.model'] = label_encoder6.fit_transform(df['make.model'])

# Ver as classes do LabelEncoder:
label_encoder1.classes_
label_encoder2.classes_
label_encoder3.classes_
label_encoder4.classes_
label_encoder5.classes_
label_encoder6.classes_

# Segregar num objeto a coluna overage, que diz se o carro passou de 90 dias na concessionária.
y = df['overage']

# Apagar as colunas:
X = df.drop(['lot.sale.days','overage'], axis=1)


# Separar os dados entre treino e teste:
from sklearn.model_selection import train_test_split

# 75% para treino e 25% para teste.
# X = Catacterísticas
# Y = Respostas
# Já separado de forma aleatória.
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y)

X_treino.count()
X_teste.count()
y_treino.count()
y_teste.count()

X_treino.count() + X_teste.count() # Total de dados.


# Aplicando Machine Learning com Regressão Linear:
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error


# Aplicando Machine Learning com Árvode de decisão:
from sklearn import tree

# Classificador do tipo arvore de decisão
# Instanciar o objeto com o classificador
arvore = tree.DecisionTreeClassifier()

# Treinar o algorítimo:
arvore.fit(X_treino, y_treino)

# Validação do Modelo:
# Passar os dados de teste para o algorítimo com os dados que ele não conhece.
# Vai retornar uma classificação: se é overage ou não.
# Essa é a predição.
# y_teste é o valor real do X_teste.
resultado = arvore.predict(X_teste)
resultado

# Mensurar a quantidade de acertos do modelo:
from sklearn import metrics

print(metrics.classification_report(y_teste, resultado))
# Precisão com "Não" o algorítimo acertou 64%
# Precisão com "Sim" o algorítimo acertou 58%
# Em média o modelo está acertando 61 % das vezes.

# O modelo pode melhorar com engenharia de features e tunning dos parâmetros.

# Ver as features mais imporantes para o modelo:
arvore.feature_importances_
features_imp = pd.Series(arvore.feature_importances_, index=X_treino.columns).sort_values(ascending=False)

sns.barplot(x=features_imp, y=features_imp.index)
plt.xlabel('Importância de features')
plt.ylabel('Features')
plt.title('Importância de features')
plt.show()

# Features não importantes podem ser removidas e o modelo rodado novamente, para testar se o modelo se sai melhor ou não.


# Visualizar a árvore de decisão:

import pydot
import graphviz
from ipywidgets import interactive

dot_data = tree.export_graphviz(
         arvore,
         feature_names=X.columns, # Nome das colunas
         class_names=y, # Classificação
         max_depth=3, # ver apenas 3 níveis de nós / ramos
         filled=True,
         rounded=True, # Caixinhas arredondadas
         node_ids=True, # Mostrar o número do nó
         label='all',
        )
# Node 0: é o nó mais importante para a árvove
graph = graphviz.Source(dot_data)
graph



## Outros plots de diagramas:
# https://graphviz.readthedocs.io/en/stable/manual.html
from graphviz import Graph
g = Graph(format='png')
graph.render('~/aphonso/Ap/DSBD_Git/Data Science do zero - Aula ML/graph', view=True)

# Diagrama:
d = graphviz.Digraph()
d.edge('Hello', 'World')
d