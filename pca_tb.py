import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

#Estilo de graficos
plt.style.use('ggplot')
#proceso analogo de preparacion de data que en VHI
data = pd.read_csv('data/TB.csv',sep=',',thousands=',', index_col = 0)
data.index.names = ['country']
data.columns.names = ['year']
X = data.ix[:,'1990':'2007'].values
X_std = StandardScaler().fit_transform(X)

#vector de medias
mean_vec = np.mean(X_std, axis=0)
#matriz de covarianza
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
#pares propios, se ordenan de mayor a menor
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs.sort()
eig_pairs.reverse()	

#4 pares principales
tot = sum(eig_vals)
var_exp=[]
for i in range(0,4):
	var_exp.append((100/tot)*sorted(eig_vals, reverse=True)[i])

cum_var_exp = np.cumsum(var_exp)
#grafico varianzas
plt.figure(figsize=(6, 4))
plt.bar(range(4), var_exp, alpha=0.5, align='center',
        label='% Individual de Varianza Descrita')
plt.step(range(4), cum_var_exp, where='mid',
         label='% Acumulado de Varianza Descrita')
plt.ylabel('Radio de Varianza Explicada')
plt.xlabel('Componentes Principales')
plt.legend(loc='best')
plt.tight_layout()

#se utilizan las dos primeras pc's, se proyectan y se genera la muestra 2d
matrix_w = np.hstack((eig_pairs[0][1].reshape(18,1),
                      eig_pairs[1][1].reshape(18,1)))
#proyeccion de las 2 pc's 
Y_sklearn = X_std.dot(matrix_w)
data_2d = pd.DataFrame(Y_sklearn)
data_2d.index = data.index
data_2d.columns = ['PC1','PC2']

#la e
#media y varianzas de data 2d
row_means = data.mean(axis=1)
row_trends = data.diff(axis=1).mean(axis=1)
data_2d.plot(kind='scatter', x='PC1', y='PC2', figsize=(16,8), c=row_means,cmap='Blues')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')

#scatter diver
data_2d.plot(kind='scatter', x='PC1', y='PC2', figsize=(16,8), c=row_means,cmap='seismic')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')

#grafico burbujas
data_2d.plot(kind='scatter', x='PC1', y='PC2', figsize=(16,8), s=10*row_means, c=row_means,cmap='RdBu')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
#grafico burbujas etiquetado
fig, ax = plt.subplots(figsize=(16,8))
row_means = data.mean(axis=1)
row_trends = data.diff(axis=1).mean(axis=1)
data_2d.plot(kind='scatter', x='PC1', y='PC2', ax=ax, s=10*row_means, c=row_means,cmap='RdBu')
Q3_TB_world = data.mean(axis=1).quantile(q=0.85)
TB_country = data.mean(axis=1)
names = data.index
for i, txt in enumerate(names):
	if(TB_country[i]>Q3_TB_world):
		ax.annotate(txt, (data_2d.iloc[i].PC1+0.2,data_2d.iloc[i].PC2))
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()
