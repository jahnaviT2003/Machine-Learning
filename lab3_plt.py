import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
#EXTRACTING THE DATA FROM THE CSV FILE
df=pd.read_csv('C:\VIT\VS CODE\WIN SEMESTER 2021-2022\ML_LAB\heart.csv')
#print(df)

#HISTOGRAM
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.hist(df['age'],bins=7)
plt.title('HEART DATA')
plt.xlabel('age')
plt.ylabel('trestbps')
plt.show()

#LM PLOT
sns.set(style='whitegrid')
tips = sns.load_dataset("tips")
#print(tips.head())
g=sns.lmplot(x="tip", y="total_bill", data=tips,aspect=2)
plt.title("TIP DATA")
plt.show()

#BOXPLOT
sns.boxplot(x="sex", y="age", data=df)
sns.boxplot(data=df,orient="h")
plt.show()
dataset = datasets.load_iris()
feature_columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']
X = df[feature_columns] # Features
y = df.target # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
