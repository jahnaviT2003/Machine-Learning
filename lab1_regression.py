#-----------SIMPLE LINEAR REGRESSION-----------#
#IMPORTING SOME PYTHON LIBRARIES 
import pandas as pd
import matplotlib.pyplot as plt
import math

#READING AND PRINTING THE DATA FROM THE CSV FILE
data = pd.read_csv('C:\VIT\VS CODE\WIN SEMESTER 2021-2022\ML_LAB\sample_data.csv')
#print(data)

#RELATIONSHIP BETWEEN INCOME AND EXPENDITURE
data.plot(kind='scatter', x='income', y='expenditure', color='green', alpha=0.5, figsize=(8, 5))
plt.title('Relationship between Income and Expenditure', size=16)
plt.xlabel('INCOME (RUPEES)', size=14)
plt.ylabel('EXPENDITURE (RUPEES)', size=14)
#plt.show()

#FINDING THE SQUARE VALUES OF INCOME AND EXPENDITURE
data['income_square']=data['income']**2
data['expenditure_square']=data['expenditure']**2
#MULTIPLYING INCOME AND EXPENDITURE
data['income_expenditure']=data['income']*data['expenditure']
#print(data)

#LINEAR REGRESSION CALCULATION 
income_size=data['income'].size
E1= (sum(data['income_expenditure']) - sum(data['income'])*sum(data['expenditure']/income_size)) / (sum(data['income_square'])-(sum(data['income'])**2)/income_size)
#print("Slope of the Regression Line : \n",round(E1,2))
E0 = (sum(data['expenditure'])-(E1*sum(data['income'])))/income_size
#print("Intercept of the Regression Line : \n",round(E0,2))

#PREDICTED EXPENDITURE
data['predicted_expenditure']=(E0+E1*data['income'])
#print(data)

#REGRESSION PLOT FOR INCOME VS EXPENDITURE
data.plot(kind='scatter', x='income', y='expenditure', color='red', alpha=0.5, figsize=(10, 7))
plt.plot(data.income, data.predicted_expenditure, color='blue', linewidth=2.5)
plt.text(35000, 90000, 'y={:.2f}+{:.2f}*x'.format(E0,E1), color='orange', size=14)
plt.title('Regression Plot', size=16)
plt.xlabel('INCOME (RUPEES)', size=14)
plt.ylabel('EXPENDITURE (RUPEES)', size=14)
#plt.show()

#CALCULATING THE RESIDUALS
data['Residuals']=(data['expenditure']-data['predicted_expenditure'])
#print(data)

#PLOT OF THE INCOME VS RESIDUALS
data.plot(kind='scatter', x='income', y='Residuals', color='deeppink', alpha=0.95, figsize=(10, 7))
plt.hlines(y=0,xmin=10000,xmax=145000, color='darkblue', linewidth=2,linestyles="--")
plt.title('RESIDUALS PLOT', size=16)
plt.xlabel('INCOME (RUPEES)', size=14)
plt.ylabel('EXPENDITURE (RUPEES)', size=14)
plt.legend()
plt.show()

