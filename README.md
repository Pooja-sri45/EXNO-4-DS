# EXNO:4-DS
## NAME:POOJASRI L
## REG.NO:212223220076
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("income(1) (1).csv",na_values=[ " ?"])
data
```
<img width="1505" height="647" alt="image" src="https://github.com/user-attachments/assets/8089faca-dd69-4139-a648-b85460f4947e" />

```
data.isnull().sum()
```
<img width="322" height="542" alt="image" src="https://github.com/user-attachments/assets/bec7c99c-4ec9-42dd-800b-1776514ac68e" />

```
missing=data[data.isnull().any(axis=1)]
missing
```
<img width="1467" height="670" alt="image" src="https://github.com/user-attachments/assets/a357a676-3f68-49b7-a4c1-be1b1c59d2a5" />

```
data2=data.dropna(axis=0)
data2
```
<img width="1477" height="625" alt="image" src="https://github.com/user-attachments/assets/0e8f5c0b-d219-4055-b8ed-e61c4f087889" />

```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
<img width="547" height="242" alt="image" src="https://github.com/user-attachments/assets/197d344e-38f9-41a0-bf4b-dbc1b04711ea" />

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
<img width="361" height="462" alt="image" src="https://github.com/user-attachments/assets/bb347c92-050d-4206-92d4-e0f966b80509" />

```
data2
```
<img width="1397" height="477" alt="image" src="https://github.com/user-attachments/assets/4aaa2821-260c-4712-9798-ef1acd184dd7" />

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
<img width="1420" height="536" alt="image" src="https://github.com/user-attachments/assets/6b8a9186-ac78-42ef-9041-aa381f0bb10c" />
<img width="1406" height="502" alt="image" src="https://github.com/user-attachments/assets/629d2acf-ae2e-49cd-9d13-73b5ff23ba36" />

```
columns_list=list(new_data.columns)
print(columns_list)
```

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
<img width="1508" height="47" alt="image" src="https://github.com/user-attachments/assets/65a96554-efb8-4179-b756-9b374b0f937e" />

```
y=new_data['SalStat'].values
print(y)
```
<img width="182" height="40" alt="image" src="https://github.com/user-attachments/assets/c9c09b88-5cd7-4934-8ff0-9e7654549563" />

```
x=new_data[features].values
print(x)
```
<img width="472" height="143" alt="image" src="https://github.com/user-attachments/assets/ead54da1-0f75-4ec9-a9c2-934c05312ab8" />

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
<img width="310" height="82" alt="image" src="https://github.com/user-attachments/assets/d1db62c6-7c9b-4d03-94f6-fc7de2ceb6b2" />

```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
<img width="185" height="60" alt="image" src="https://github.com/user-attachments/assets/8bc143ee-3b3c-4fff-bc36-69a90eb2c551" />

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
<img width="252" height="40" alt="image" src="https://github.com/user-attachments/assets/157ace7e-ebbd-41b5-8f95-ed100b433090" />

```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
<img width="288" height="40" alt="image" src="https://github.com/user-attachments/assets/433a0b4c-099c-4b8b-8957-8380e824fe9e" />

```
data.shape
```
<img width="185" height="38" alt="image" src="https://github.com/user-attachments/assets/b2a2aa04-f7ad-4623-bbca-054b7e83b32e" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
'Feature1': [1,2,3,4,5],
'Feature2': ['A','B','C','A','B'],
'Feature3': [0,1,1,0,1],
'Target' : [0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
<img width="331" height="42" alt="image" src="https://github.com/user-attachments/assets/809e3e60-9b68-42bb-b206-0a6526f2d71e" />


```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
<img width="486" height="231" alt="image" src="https://github.com/user-attachments/assets/a346499f-be1e-4615-9c24-d78354ee3288" />

```
tips.time.unique()
```
<img width="472" height="75" alt="image" src="https://github.com/user-attachments/assets/871daced-bacf-45f5-bf27-4b520d8f10d9" />

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
<img width="251" height="97" alt="image" src="https://github.com/user-attachments/assets/40036a2f-e1bf-45a6-813b-38b88834bf5d" />

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
<img width="395" height="62" alt="image" src="https://github.com/user-attachments/assets/ef45177c-5ef9-433a-9ac8-e17e9020ad07" />

# RESULT:
       # INCLUDE YOUR RESULT HERE
