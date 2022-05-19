# EX-06-Feature-Transformation

## AIM
To Perform the various feature transformation techniques on a dataset and save the data to a file. 

# Explanation
Feature Transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

 
# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Transformation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE
### Data To Transform
```
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
import statsmodels.api as sm  
import scipy.stats as stats 
df=pd.read_csv("Data_to_Transform.csv")
df
df1=df.copy()
# log transformation
df1["ModeratePositiveSkew_log"] = np.log(df1.ModeratePositiveSkew)
df1.ModeratePositiveSkew_log
# reciprocal transformation
df1["HighlyPositiveSkew_recip"] = 1/df.HighlyPositiveSkew
df1.HighlyPositiveSkew_recip
#square transformation
df1["HighlyNegativeSkew_square"]= df1.HighlyNegativeSkew**(1/1.2)
df1.HighlyNegativeSkew_square
# square root transformation
df1['ModeratePositiveSkew_sqrt'] = np.sqrt(df.ModeratePositiveSkew)
df1.ModeratePositiveSkew_sqrt
# boxcox transforms
df1["HighlyPositiveSkew_boxcox"], parameters=stats.boxcox(df1.HighlyPositiveSkew)
df1.HighlyPositiveSkew_boxcox
df1["HighlyNegativeSkew_yeojohnson"], parameters=stats.yeojohnson(df1.HighlyNegativeSkew)
df1.HighlyNegativeSkew_yeojohnson#QUANTILE TRANSFORMATION:  
from sklearn.preprocessing import QuantileTransformer   
qt=QuantileTransformer(output_distribution='normal') 
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["ModerateNegativeSkew"]])  
sm.qqplot(df['ModerateNegativeSkew'],line='45')  
plt.show()
sm.qqplot(df['Moderate Negative Skew_1'],line='45')  
plt.show()  
```
### titanic_dataset
```
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
import statsmodels.api as sm  
import scipy.stats as stats
df=pd.read_csv("titanic_dataset.csv")  
df 
df.drop("Name",axis=1,inplace=True)  
df.drop("Cabin",axis=1,inplace=True)  
df.drop("Ticket",axis=1,inplace=True)  
df.isnull().sum()
df["Age"]=df["Age"].fillna(df["Age"].median())  
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])  
df.info()  
from sklearn.preprocessing import OrdinalEncoder
embark=["C","S","Q"]  
emb=OrdinalEncoder(categories=[embark])  
df["Embarked"]=emb.fit_transform(df[["Embarked"]])  
df
#FUNCTION TRANSFORMATION:  
#Log Transformation  
np.log(df["Fare"]) 
#ReciprocalTransformation 
np.reciprocal(df["Age"])
#Squareroot Transformation:  
np.sqrt(df["Embarked"])
#POWER TRANSFORMATION:  
df["Age _boxcox"], parameters=stats.boxcox(df["Age"])  
df  
df["Pclass _boxcox"], parameters=stats.boxcox(df["Pclass"])    
df 
df["Fare _yeojohnson"], parameters=stats.yeojohnson(df["Fare"])  
df 
df["SibSp _yeojohnson"], parameters=stats.yeojohnson(df["SibSp"])  
df  
df["Parch _yeojohnson"], parameters=stats.yeojohnson(df["Parch"])  
df
#QUANTILE TRANSFORMATION  
from sklearn.preprocessing import QuantileTransformer   
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Age_1"]=qt.fit_transform(df[["Age"]])  
sm.qqplot(df['Age'],line='45')  
plt.show() 
sm.qqplot(df['Age_1'],line='45')  
plt.show()  
df["Fare_1"]=qt.fit_transform(df[["Fare"]])  
sm.qqplot(df["Fare"],line='45')  
plt.show()  
sm.qqplot(df['Fare_1'],line='45')  
plt.show()
df.skew()
df  
```

# OUPUT
## Data to Transform
![op1_1](https://user-images.githubusercontent.com/95342910/169196223-cbd25a14-3edb-4b47-8e15-6b465f7ae195.png)
### log transformation
![op1_2](https://user-images.githubusercontent.com/95342910/169196245-f6ca1b35-7795-4a54-9637-c35ef9f89159.png)
### reciprocal transformation
![op1_3](https://user-images.githubusercontent.com/95342910/169196268-2bbbb854-4603-4ce2-8fb1-e5fac00814fe.png)
### square transformation
![op1_4](https://user-images.githubusercontent.com/95342910/169196284-67c490cc-3520-4ec6-962e-2ae7eaf667f2.png)
### squareroot transformation
![op1_5](https://user-images.githubusercontent.com/95342910/169196318-1f58e241-e84b-4960-926e-6ba220b58e5e.png)
### boxcox transformation
![op1_6](https://user-images.githubusercontent.com/95342910/169196337-bcc0985d-128d-4d52-af51-c7c4642c1292.png)
### yeojohnson transformation
![op1_7](https://user-images.githubusercontent.com/95342910/169196357-094b3c7b-853f-4fad-849e-9fbfb8ceecf5.png)
### quantile transformation
![op1_8](https://user-images.githubusercontent.com/95342910/169196380-5d387e97-5258-4326-947d-35d1545f7e2d.png)
![op1_9](https://user-images.githubusercontent.com/95342910/169196400-f4dd922f-f99f-4288-b69c-54e0f3327884.png)
## titanic_dataset
![op2_1](https://user-images.githubusercontent.com/95342910/169196421-ab1b2fc3-9e70-4eb8-90bb-e527f0c768b9.png)
![op2_2](https://user-images.githubusercontent.com/95342910/169196430-f2f7d5d8-4adf-4d47-b6ea-26e029fd997f.png)
![op2_3](https://user-images.githubusercontent.com/95342910/169196440-5434af27-9b05-4617-8a9f-516597e938e5.png)
![op2_4](https://user-images.githubusercontent.com/95342910/169196445-ef40e45d-d18c-4bc0-b0f3-d796a8cfd407.png)
### Log transformation
![op2_5](https://user-images.githubusercontent.com/95342910/169196471-7b444dfb-8609-427e-abac-f32dacdc6ebc.png)
### reciprocal transformation
![op2_6](https://user-images.githubusercontent.com/95342910/169196489-ccc8926f-10ff-475e-8e5a-25c878d495b6.png)
### squareroot transformation
![op2_7](https://user-images.githubusercontent.com/95342910/169196503-f45f3e9f-82ad-48a7-8f58-32311850099c.png)
### boxcox transformation
![op2_8](https://user-images.githubusercontent.com/95342910/169196521-005a1e9a-c68a-4c87-8d5b-0fc8523fe5c8.png)
![op2_9](https://user-images.githubusercontent.com/95342910/169196536-fe049a2d-46f6-4dfb-a1c8-bee1db084f21.png)
### yeojohnson  transformation
![op2_10](https://user-images.githubusercontent.com/95342910/169196555-06e5be2c-e68e-4f62-89cf-71fd3406e63f.png)
![op2_11](https://user-images.githubusercontent.com/95342910/169196563-694ef7fe-ddd1-427a-80c3-eaa6eeb6551b.png)
![op2_12](https://user-images.githubusercontent.com/95342910/169196570-898d753d-b774-4bb3-8046-7072410f12ba.png)
### QUANTILE TRANSFORMATION 
![op2_13](https://user-images.githubusercontent.com/95342910/169196591-85cf3e23-fe28-406a-b9c3-f612943b592f.png)
![op2_14](https://user-images.githubusercontent.com/95342910/169196603-f5b7adde-f543-44ce-8b7d-161b02be2f39.png)

![op2_15](https://user-images.githubusercontent.com/95342910/169196619-565dc429-12b7-4692-a899-9cc05ae304fa.png)
![op2_16](https://user-images.githubusercontent.com/95342910/169196628-eeb2789f-56d6-443b-bd3c-50b998254973.png)

# RESULT
Hence, Feature transformation techniques is been performed on given dataset and saved into a file successfully.

