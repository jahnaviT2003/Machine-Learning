import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from pyparsing import col

#READING AND PRINTING THE DATA FROM THE CSV FILE
fish_data = pd.read_csv('C:\VIT\VS CODE\WIN SEMESTER 2021-2022\ML_LAB\Fish.csv')
print(fish_data)

# CHECKING FOR NULL VALUES 
fish_data.info()

# CHECKING FOR OUTLIERS
fish_data.describe()

#CONVERTING CATEGORICAL VARIABLES TO NUMERIC VARIABLES

#CONVERTING SPECIES CATEGORICAL TO NUMERIC VARIABLE
fish_data['Species'].replace(['Bream', 'Roach', 'Whitefish', 'Parkki', 'Perch', 'Pike', 'Smelt'], [1, 2, 3, 4, 5, 6, 7], inplace=True)
#UPDATED DATAFRAME
print(fish_data)

from sklearn.model_selection import train_test_split

np.random.seed(0)
df_train, df_test = train_test_split(fish_data, train_size = 0.7, test_size = 0.3, random_state = 100)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

num_vars = ['Length1', 'Length2', 'Length3', 'Height', 'Width', 'Weight', 'Species']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
print(df_train)

# Dividing the training data set into X and Y
y_train = df_train.pop('Weight')
X_train = df_train

#Build a linear model

import statsmodels.api as sm
X_train_lm = sm.add_constant(X_train)

linear_model = sm.OLS(y_train, X_train_lm).fit()
print(linear_model.summary())

# Checking for the VIF values of the variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Creating a dataframe that will contain the names of all the feature variables and their VIFs
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
#print(vif)

# Dropping highly correlated variables and insignificant variables
X = X_train.drop('Species', 1)
# Build a fitted model after dropping the variable
X_train_lm = sm.add_constant(X)
lr_2 = sm.OLS(y_train, X_train_lm).fit()
# Printing the summary of the model
print(lr_2.summary())

# Calculating the VIFs again for the new 

vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
print(vif)

#Residual Analysis of Train Data
import seaborn as sb
y_train_Weight = lr_2.predict(X_train_lm)
fig = plt.figure()
sb.displot((y_train - y_train_Weight), bins = 20)
#plt.show()

df_test[num_vars] = scaler.transform(df_test[num_vars])
#print(df_test)
y_test = df_test.pop('Weight')
X_test = df_test

# Adding constant variable to test dataframe
X_test_lm = sm.add_constant(X_test)

# Creating X_test_m4 dataframe by dropping variables from X_test_m4
X_test_lm = X_test_lm.drop(["Species"], axis = 1)

# Making predictions using the final model
y_pred = lr_2.predict(X_test_lm)

from sklearn.metrics import r2_score
r2_score(y_true = y_test, y_pred = y_pred)

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
rfe = RFE(lm, 10)
rfe = rfe.fit(X_train, y_train)
print(list(zip(X_train.columns, rfe.support_, rfe.ranking_)))

# Creating X_test dataframe with RFE selected variables

X_train_new = X_train.drop(["Species"], axis = 1)

# Adding a constant variable 
import statsmodels.api as sm  
X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model

#print(lm.summary())


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
#print(vif)

y_train_Weight = lm.predict(X_train_lm)
# Importing the required libraries for plots.
import seaborn as sns

# Plot the histogram of the error terms
fig = plt.figure()
sns.displot((y_train - y_train_Weight), bins = 20)
plt.show()

y_pred_mlr= lm.predict(X_test)
#Predicted values
print("Prediction for test set: {}".format(y_pred_mlr))

lm_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
print(lm_diff.head())

from sklearn import metrics
meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))

print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)