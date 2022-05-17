import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as mt
from sklearn import metrics
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

#EXTRACTING DATA FROM CSV FILE
data = pd.read_csv('C:\VIT\VS CODE\WIN SEMESTER 2021-2022\ML_LAB\hgtwgt_sex.csv')
#DISPLAYS THE DATAFRAME OF DATASET
print(data)

#DESCRIBES ABOUT THE DATASET
print(data.describe())

#CONVERTING THE CATEGORICAL VARIABLES TO NUMERIC VARIABLES
df = pd.DataFrame(data)
df = df.replace('Male', '0')
df = df.replace('Female', '1')
print(df)

#SPLITTING DATA INTO TESTDATA AND TRAIN DATA
np.random.seed(0) 
df_train, df_test = train_test_split(df, train_size=0.7, test_size=0.3, random_state=100)
scaler = preprocessing.MinMaxScaler()
variables = ['Height','Weight','Sex','Index']
train_x_new = scaler.fit_transform(df_train)
pd.DataFrame(train_x_new)
print(train_x_new)

#K NEAREST NEIGHBOURS FOR TRAIN DATA
y = df_train['Index']
x = df_train
n = KNeighborsClassifier(n_neighbors=7)
n.fit(x, y)
print(n.get_params(deep=True))
print(n.kneighbors(X=None, n_neighbors=None, return_distance=True))

y_pred = n.predict(x)
print(y_pred)
print(y)

mat = confusion_matrix(y, y_pred)
sns.heatmap(mat, annot=True, fmt="d")
print(classification_report(y, y_pred))

#K NEAREST NEIGHBOURS FOR TEST DATA
ytest = df_test['Index']
xtest = df_test
nn = KNeighborsClassifier(n_neighbors=7)
nn.fit(xtest, ytest)
print(nn.get_params(deep=True))
print(nn.kneighbors(X=None, n_neighbors=None, return_distance=True))

yy_pred = nn.predict(xtest)
print(yy_pred)
print(ytest)

matrix = confusion_matrix(ytest, yy_pred)
sns.heatmap(matrix, annot=True, fmt="d")
print(classification_report(ytest, yy_pred))


