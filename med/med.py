import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

medical_df = pd.read_csv('/Users/riteshkumar/Downloads/ML projects/med/insurance.csv')

medical_df.head()

medical_df.shape
medical_df.info()
medical_df.describe()
plt.figure(figsize=(3,3))
sns.displot(data=medical_df,x='age')

plt.figure(figsize=(3,3))
sns.displot(data=medical_df,x='sex',kind='hist')

medical_df['sex'].value_counts()
medical_df.info()

plt.figure(figsize=(4,4))
sns.displot(data=medical_df,x='bmi')
plt.show()

medical_df['bmi'].value_counts()
plt.figure(figsize=(4,4))
sns.countplot(medical_df['children'])
plt.show()

medical_df['children'].value_counts()

plt.figure(figsize=(4,4))
sns.countplot(data=medical_df,x='smoker')
plt.show()

medical_df.head()
medical_df['region'].value_counts()


medical_df.replace({'sex':{'male':0,'female':1}},inplace=True)
medical_df.replace({'smoker':{'yes':0,'no':1}},inplace=True)
medical_df.replace({'region':{'southeast':0,'southwest':1,'northwest':2,'northeast':3}},inplace=True)

medical_df.head()

X = medical_df.drop('charges',axis=1)
y = medical_df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

X_train.shape
X_test.shape
lg = LinearRegression()
lg.fit(X_train,y_train) # 80 model will be train
y_pred = lg.predict(X_test) # 10 model will be predicted

r2_score(y_test,y_pred)

input_df = ()
np_df = np.asarray(input_df)
input_df_reshaped = np_df.reshape(1,-1)
prediction = lg.predict(input_df_reshaped)
print("")