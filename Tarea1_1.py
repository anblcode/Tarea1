import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import math
plt.style.use('ggplot')

data = pd.read_csv(
'data/titanic-train.csv',
sep=';')


print data.shape
print data.info()
print data.describe()


print data.tail()
print data.head()
print data[200:210][:]
print data[['Sex','Survived']].tail()
print data[['Sex','Survived']].head()
print data[['Fare']][200:210]


print data['Sex'].value_counts()
print data.groupby('Sex').Survived.count()
print data.groupby('Sex').Survived.mean()
print data.groupby('Survived')['Sex'].value_counts()
print data.groupby('Sex')['Survived'].value_counts()
axes = data.groupby('Sex')['Survived'].value_counts().unstack().plot(kind='bar')
L=plt.legend(fontsize=9.5)
L.get_texts()[0].set_text('Fallecidos')
L.get_texts()[1].set_text('Sobrevivientes')
a=axes.get_xticks().tolist()
a[0]='Mujeres'
a[1]='Hombres'
axes.set_xticklabels(a, rotation=0)
grouped_props = data.groupby('Survived')['Sex'].value_counts()
print data.groupby('Survived').size()
axes = grouped_props.unstack().plot(kind='bar')
L=plt.legend()
L.get_texts()[0].set_text('Mujeres')
L.get_texts()[1].set_text('Hombres')
a=axes.get_xticks().tolist()
a[0]='Fallecidos'
a[1]='Sobrevivientes'
axes.set_xticklabels(a, rotation=0)



print data.groupby('Survived')['Age'].mean()
lebs = data.boxplot(column='Age',by='Survived')
la=lebs.get_xticks().tolist()
la[0]='Fallecidos'
la[1]='Sobrevivientes'
lebs.set_xticklabels(la, rotation=0)
plt.suptitle("Boxplot agrupado por sobrevivencia")
plt.title('Edad');

lebs = data.hist(column='Age',by='Survived')

print 'Fallecidos con edad nula: ' + str(sum(data[data.Survived==0]['Age'].isnull()))
print 'Fallecidos con edad NO nula: ' + str(sum(data[data.Survived==0]['Age'].notnull()))
print 'Sobrevivientes con edad nula: ' + str(sum(data[data.Survived==1]['Age'].isnull()))
print 'Sobrevivientes con edad NO nula: ' + str(sum(data[data.Survived==1]['Age'].notnull()))
print data[data.Age==data['Age'].max()]



median_male = round(data[(data.Age.notnull()) & (data.Sex=='male')]['Age'].mean(),0)
median_female = round(data[(data.Age.notnull()) & (data.Sex=='female')]['Age'].mean(), 0)
#La siguiente instruccion no funciona ya que estoy accediendo a la informacion
#que ahi hay, como se hace anteriormente, sirve para comparar y no para cambiar 
#su valor.
#data.loc[(data.Age.isnull()) & (data.Sex=='female')]['Age'] = median_female
data.loc[(data.Age.isnull()) & (data.Sex=='female'), 'Age'] = median_female
data.loc[(data.Age.isnull()) & (data.Sex=='male'), 'Age'] = median_male
#print data[['Age','Sex']][200:230]

print data.groupby('Survived')['Age'].mean()
lebs = data.boxplot(column='Age',by='Survived')
la=lebs.get_xticks().tolist()
la[0]='Fallecidos'
la[1]='Sobrevivientes'
lebs.set_xticklabels(la, rotation=0)
plt.suptitle("Boxplot agrupado por sobrevivencia")
plt.title('Edad');

lebs = data.hist(column='Age',by='Survived')

print 'Fallecidos con edad nula: ' + str(sum(data[data.Survived==0]['Age'].isnull()))
print 'Fallecidos con edad NO nula: ' + str(sum(data[data.Survived==0]['Age'].notnull()))
print 'Sobrevivientes con edad nula: ' + str(sum(data[data.Survived==1]['Age'].isnull()))
print 'Sobrevivientes con edad NO nula: ' + str(sum(data[data.Survived==1]['Age'].notnull()))
print data[data.Age==data['Age'].max()]



print data['Pclass'].unique()
print data.groupby(['Pclass']).groups.keys()
print data.groupby(['Survived', 'Pclass']).size()/data.groupby(['Survived']).size()
print data.groupby(['Pclass','Survived']).size()/data.groupby(['Pclass']).size()
lebs = data.groupby('Pclass')['Survived'].value_counts().unstack().plot(kind='bar')
la=lebs.get_xticks().tolist()
la[0]='1era Clase'
la[1]='2da Clase'
la[2]='3era Clase'
lebs.set_xticklabels(la, rotation=0)
L=plt.legend(fontsize=9)
L.get_texts()[0].set_text('Fallecidos')
L.get_texts()[1].set_text('Sobrevivientes')
lebs = data.groupby('Survived')['Pclass'].value_counts().unstack().plot(kind='bar')
la=lebs.get_xticks().tolist()
la[0]='Fallecidos'
la[1]='Sobrevivientes'
lebs.set_xticklabels(la, rotation=0)
L=plt.legend()
L.get_texts()[0].set_text('1era Clase')
L.get_texts()[1].set_text('2da Clase')
L.get_texts()[2].set_text('3era Clase')
females = data[data.Sex == 'female'].groupby(['Survived','Pclass']).size()/data[data.Sex == 'female'].groupby(['Survived']).size()
print females
lebs = females.unstack().plot(kind='bar')
la=lebs.get_xticks().tolist()
la[0]='Fallecidos'
la[1]='Sobrevivientes'
lebs.set_xticklabels(la, rotation=0)
L=plt.legend()
L.get_texts()[0].set_text('1era Clase')
L.get_texts()[1].set_text('2da Clase')
L.get_texts()[2].set_text('3era Clase')

males = data[data.Sex == 'male'].groupby(['Survived','Pclass']).size()/data[data.Sex == 'male'].groupby(['Survived']).size()
print males
lebs = males.unstack().plot(kind='bar')
la=lebs.get_xticks().tolist()
la[0]='Fallecidos'
la[1]='Sobrevivientes'
lebs.set_xticklabels(la, rotation=0)
L=plt.legend()
L.get_texts()[0].set_text('1era Clase')
L.get_texts()[1].set_text('2da Clase')
L.get_texts()[2].set_text('3era Clase')



data['prediction']=0 
data.prediction[(data.Sex == 'female') & (data.Pclass!=2)]=1
data.prediction[(data.Sex == 'male') & (data.Pclass==1)]=1
print 'Precision train: '+ str(data[data.prediction==1][data.Survived==1].size/float(data[data.prediction==1].size))
print 'Recall train: ' + str(data[data.prediction==1][data.Survived==1].size/float(data[data.Survived==1].size))
data.to_csv('predicciones-titanic.csv')


dataFareTyp=data[data.Fare<50]
dataFareTyp.boxplot(column='Fare')
dataFareTyp.hist(column='Fare', by='Survived')
pd.options.display.mpl_style = None 
dataFareTypSurv=data[(data.Fare<63) & (data.Survived==1)]
dataFareTypDied=data[(data.Fare<63) & (data.Survived==0)]
fig, ax = plt.subplots()
sns.distplot(dataFareTypSurv['Fare'], label='Sobrevivientes')
sns.distplot(dataFareTypDied['Fare'], label='Fallecidos')



#Se hacen los 5 intervalos de precios
data['rango']=0
data.rango[(data.Fare <= 9.9)] = 1
data.rango[(data.Fare >= 10) & (data.Fare <= 19.9)] = 2
data.rango[(data.Fare >= 20) & (data.Fare <= 29.9)] = 3
data.rango[(data.Fare >= 30) & (data.Fare <= 39.9)] = 4
data.rango[(data.Fare >= 40)] = 5

data['prediction']=0 
data.prediction[(data.Sex == 'female') & (data.Pclass!=2) & (data.rango!=2)]=1
data.prediction[(data.Sex == 'male') & (data.Pclass==1) & (data.rango!=2)]=1
print 'Precision train: '+ str(data[data.prediction==1][data.Survived==1].size/float(data[data.prediction==1].size))
print 'Recall train: ' + str(data[data.prediction==1][data.Survived==1].size/float(data[data.Survived==1].size))


data1 = pd.read_csv(
'data/titanic-test.csv',
sep=',')
data2 = pd.read_csv(
'data/gendermodel.csv',
sep=',')

#Se hacen los 5 intervalos de precios
data1['rango']=0
data1.rango[(data1.Fare <= 9.9)] = 1
data1.rango[(data1.Fare >= 10) & (data1.Fare <= 19.9)] = 2
data1.rango[(data1.Fare >= 20) & (data1.Fare <= 29.9)] = 3
data1.rango[(data1.Fare >= 30) & (data1.Fare <= 39.9)] = 4
data1.rango[(data1.Fare >= 40)] = 5

data1['Survived']=data2['Survived']

data1['prediction']=0 
data1.prediction[(data1.Sex == 'female') & (data1.Pclass!=2)]=1
data1.prediction[(data1.Sex == 'male') & (data1.Pclass==1)]=1
print 'Precision (h) test: ' + str(data1[data1.prediction==1][data1.Survived==1].size/float(data1[data1.prediction==1].size))
print 'Recall (h) test: ' + str(data1[data1.Survived==1][data1.prediction==1].size/float(data1[data1.Survived==1].size))

data1['prediction']=0 
data1.prediction[(data1.Sex == 'female') & (data1.Pclass!=2) & (data1.rango!=2)]=1
data1.prediction[(data1.Sex == 'male') & (data1.Pclass==1) & (data1.rango!=2)]=1
print 'Precision (i) train: '+ str(data1[data1.prediction==1][data1.Survived==1].size/float(data1[data1.prediction==1].size))
print 'Recall (i) train: ' + str(data1[data1.prediction==1][data1.Survived==1].size/float(data1[data1.Survived==1].size))

plt.show()