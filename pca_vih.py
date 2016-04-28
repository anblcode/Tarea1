import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

#Estilo de graficos
plt.style.use('ggplot')
#Lee el archivo, separa por ; y reconoce decimal con ,
#Se eliminan los anos con datos faltantes, y paises sin datos
data = pd.read_csv('data/HIV.csv',sep=';',decimal=',',index_col=0)
data = data.drop(data.columns[range(0,11)], axis=1)
data = data[data['2010'].notnull()]
data = data.dropna()
#se nombran las filas y columnas con su 
data.shape
data.index.names = ['country']
data.columns.names = ['year']

#grafico de lineas
fig, ax = plt.subplots(figsize=(8,3))
data.loc[['Zimbabwe','Zambia','South Africa','Botswana','Colombia','Jamaica','France','Italy','United States'],'1990':].T.plot(ax=ax)
ax.legend(loc='upper left', bbox_to_anchor=(0.1, 1.1),prop={'size':'x-small'},ncol=6)
plt.tight_layout(pad=1.5)

#Crea la matriz de datos con 22 columnas, y todos los paises no eliminados
X = data.ix[:,'1990':'2011'].values
#Se ajustan los datos a media 0 y varianza unitaria
X_std = StandardScaler().fit_transform(X)
#Vector de medias 
mean_vec = np.mean(X_std, axis=0)

#se computa la matriz de covarianza como su formula lo dice
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)

#se obtienen los pares propios desde matriz cov
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

#Ordena los pares propios de mayor a menor
eig_pairs.sort()
eig_pairs.reverse()

# 4 pares principales
tot = sum(eig_vals)
var_exp = []
for i in range(0,4):
	var_exp.append(100/tot*sorted(eig_vals, reverse=True)[i])
#vector varianza acumulada
cum_var_exp = np.cumsum(var_exp)

#grafico barras
plt.figure(figsize=(6, 4))
plt.bar(range(4), var_exp, alpha=0.5, align='center',
        label='% Individual de Varianza Descrita')
plt.step(range(4), cum_var_exp, where='mid',
	label='% Acumulado de Varianza Descrita')
plt.ylabel('Radio de Varianza Descrita')
plt.xlabel('Componentes Principales')
plt.legend(loc='best')
plt.tight_layout()

#se utilizan las dos componentes principales mas grandes
matrix_w = np.hstack((eig_pairs[0][1].reshape(22,1),
		      eig_pairs[1][1].reshape(22,1)))

#se genera la proyeccion 
Y_proy = X_std.dot(matrix_w)

#la d
data_2d = pd.DataFrame(Y_proy)
data_2d.index = data.index
data_2d.columns = ['PC1','PC2']


#la e
row_means = data.mean(axis=1)
row_trends = data.diff(axis=1).mean(axis=1)
	#Scatter color secuencial 
data_2d.plot(kind='scatter', x='PC1', y='PC2', figsize=(16,8), c=row_means,cmap='Blues')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
	#Scatter color Divergente 
data_2d.plot(kind='scatter', x='PC1', y='PC2', figsize=(16,8), c=row_means,cmap='seismic')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')

#f, grafico burbujas
data_2d.plot(kind='scatter', x='PC1', y='PC2', figsize=(16,8), s=10*row_means, c=row_means,cmap='RdBu')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
#e, grafico burbuja etiquetado
fig, ax = plt.subplots(figsize=(16,8))
row_means = data.mean(axis=1)
row_trends = data.diff(axis=1).mean(axis=1)
data_2d.plot(kind='scatter', x='PC1', y='PC2', ax=ax, s=10*row_means, c=row_means, cmap='RdBu')
Q3_HIV_world = data.mean(axis=1).quantile(q=0.85)
HIV_country = data.mean(axis=1)

names = data.index
for i, txt in enumerate(names):
	if(HIV_country[i]>Q3_HIV_world):
		ax.annotate(txt, (data_2d.iloc[i].PC1+0.2,data_2d.iloc[i].PC2))

plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()
