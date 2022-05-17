import pandas as pd
import numpy as np
#Importing dataset using pandas 
#Loading Data into dataframe df
df=pd.read_csv('C:\VIT\VS CODE\WIN SEMESTER 2021-2022\ML_LAB\heart.csv')
print(df)
#Understanding the axes
print(df.sum())
#Sums down the 0 axis i.e. Rows
print(df.sum(axis=0))
#Sums across 1 axis i.e. colums
print(df.sum(axis=1))

# limit which rows are read when reading in a file
df_1=pd.read_csv('C:\VIT\VS CODE\WIN SEMESTER 2021-2022\ML_LAB\heart.csv', nrows=10)        
# only read first 10 rows
print(df_1)

df_2=pd.read_csv('C:\VIT\VS CODE\WIN SEMESTER 2021-2022\ML_LAB\heart.csv', skiprows=[1,2])
# skip the first two rows of data
print(df_2)

# randomly sample a DataFrame
train = df.sample(frac=0.75) 
# will contain 75% of the rows
print(train)

test = df[~df.index.isin(train.index)] 
# will contain the other 25%)
print(test)

# change the maximum number of rows and columns printed (‘None’ means unlimited)
pd.set_option('max_rows', None) 
# default is 60 rows
print(df)

df_4=pd.set_option('max_columns', None) 
# default is 20 columns
print(df)
# reset options to defaults
pd.reset_option('max_rows')
pd.reset_option('max_columns')
print(df)
#It describes the basics of stastical models
print(df.describe())

#It displays co-variance bewteen two columns
print(df.cov())

#It displays co-relation between two coulmns
print(df.corr())

#Retruns unique values of a coulmn
print(df['sex'].unique())

#It displays last 10 rows of the dataset
print(df.tail(10))

#It displays names of the columns
print(df.columns)

#Rename method helps to rename the column of data frame 
df2=df.rename(columns={'sex':'gender'})
print(df2)
#This statement will create new dream data frame with new column name

df.rename(columns={'sex':'gender'}, inplace=True)
print(df)

#Accessing sub dataframes
print(df[['sex','age']])

#Filtering Records
print(df[df['sex']==1])
print(df[(df['age']>50) & (df['trestbps']>120)])

#Creating New Columns
df['NewColumn1']=df['sex'] #Creating a copy of existing column-2 values
print(df)

#Creates newcolumn in which sum of 2 columns values will be added and copied into 2nd new column
df['NewColumn2']=df['oldpeak']+df['slope']
print(df)

#Adding 5 to existing column(restecg) values and copied them into 3rd new column 
df['NewColumn3']=df['thalach']+5
print(df)

print("Correlation : ",df.corr())

#SOME STATISTICAL MODELS
x = np.array(df['thalach'])
print(x)
print("Mean:",np.mean(x))
print("Median:",np.median(x))
print("Standard Deviation:",np.std(x))
print("Minimum:",x.min())
print("Dimensions:",x.shape)
print("Number of records:",x.size)
