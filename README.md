# House_Prices_Advanced_Regression_Techniques_Kaggle



- We will use the [dataset available on Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)
    - It's a **competition** dataset
    - We have a database with the **description of each of the columns (data_description.txt)**


```python
# Importing pandas
import pandas as pd
```


```python
# Importing the training dataset
base = pd.read_csv('train.csv')
```


```python
# Viewing this base
base.head(3)
```





  <div id="df-c55c6873-2160-48e3-84ea-79271985d3a9" class="colab-df-container">
    <div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 81 columns</p>
</div>
    <div class="colab-df-buttons">




<div id="df-ac881925-2940-4334-a7dd-8f8be4f58a01">
  <button class="colab-df-quickchart" onclick="quickchart('df-ac881925-2940-4334-a7dd-8f8be4f58a01')"
            title="Suggest charts"
            style="display:none;">



  </button>


</div>


  </div>





```python
# Returning the base shape
base.shape
```




    (1460, 81)




```python
# And the information
base.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 81 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Id             1460 non-null   int64  
     1   MSSubClass     1460 non-null   int64  
     2   MSZoning       1460 non-null   object 
     3   LotFrontage    1201 non-null   float64
     4   LotArea        1460 non-null   int64  
     5   Street         1460 non-null   object 
     6   Alley          91 non-null     object 
     7   LotShape       1460 non-null   object 
     8   LandContour    1460 non-null   object 
     9   Utilities      1460 non-null   object 
     10  LotConfig      1460 non-null   object 
     11  LandSlope      1460 non-null   object 
     12  Neighborhood   1460 non-null   object 
     13  Condition1     1460 non-null   object 
     14  Condition2     1460 non-null   object 
     15  BldgType       1460 non-null   object 
     16  HouseStyle     1460 non-null   object 
     17  OverallQual    1460 non-null   int64  
     18  OverallCond    1460 non-null   int64  
     19  YearBuilt      1460 non-null   int64  
     20  YearRemodAdd   1460 non-null   int64  
     21  RoofStyle      1460 non-null   object 
     22  RoofMatl       1460 non-null   object 
     23  Exterior1st    1460 non-null   object 
     24  Exterior2nd    1460 non-null   object 
     25  MasVnrType     588 non-null    object 
     26  MasVnrArea     1452 non-null   float64
     27  ExterQual      1460 non-null   object 
     28  ExterCond      1460 non-null   object 
     29  Foundation     1460 non-null   object 
     30  BsmtQual       1423 non-null   object 
     31  BsmtCond       1423 non-null   object 
     32  BsmtExposure   1422 non-null   object 
     33  BsmtFinType1   1423 non-null   object 
     34  BsmtFinSF1     1460 non-null   int64  
     35  BsmtFinType2   1422 non-null   object 
     36  BsmtFinSF2     1460 non-null   int64  
     37  BsmtUnfSF      1460 non-null   int64  
     38  TotalBsmtSF    1460 non-null   int64  
     39  Heating        1460 non-null   object 
     40  HeatingQC      1460 non-null   object 
     41  CentralAir     1460 non-null   object 
     42  Electrical     1459 non-null   object 
     43  1stFlrSF       1460 non-null   int64  
     44  2ndFlrSF       1460 non-null   int64  
     45  LowQualFinSF   1460 non-null   int64  
     46  GrLivArea      1460 non-null   int64  
     47  BsmtFullBath   1460 non-null   int64  
     48  BsmtHalfBath   1460 non-null   int64  
     49  FullBath       1460 non-null   int64  
     50  HalfBath       1460 non-null   int64  
     51  BedroomAbvGr   1460 non-null   int64  
     52  KitchenAbvGr   1460 non-null   int64  
     53  KitchenQual    1460 non-null   object 
     54  TotRmsAbvGrd   1460 non-null   int64  
     55  Functional     1460 non-null   object 
     56  Fireplaces     1460 non-null   int64  
     57  FireplaceQu    770 non-null    object 
     58  GarageType     1379 non-null   object 
     59  GarageYrBlt    1379 non-null   float64
     60  GarageFinish   1379 non-null   object 
     61  GarageCars     1460 non-null   int64  
     62  GarageArea     1460 non-null   int64  
     63  GarageQual     1379 non-null   object 
     64  GarageCond     1379 non-null   object 
     65  PavedDrive     1460 non-null   object 
     66  WoodDeckSF     1460 non-null   int64  
     67  OpenPorchSF    1460 non-null   int64  
     68  EnclosedPorch  1460 non-null   int64  
     69  3SsnPorch      1460 non-null   int64  
     70  ScreenPorch    1460 non-null   int64  
     71  PoolArea       1460 non-null   int64  
     72  PoolQC         7 non-null      object 
     73  Fence          281 non-null    object 
     74  MiscFeature    54 non-null     object 
     75  MiscVal        1460 non-null   int64  
     76  MoSold         1460 non-null   int64  
     77  YrSold         1460 non-null   int64  
     78  SaleType       1460 non-null   object 
     79  SaleCondition  1460 non-null   object 
     80  SalePrice      1460 non-null   int64  
    dtypes: float64(3), int64(35), object(43)
    memory usage: 924.0+ KB
    

## Starting to explore the data


```python
# Viewing the number of empty values
(base.isnull().sum()/base.shape[0]).sort_values(ascending=False).head(20)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PoolQC</th>
      <td>0.995205</td>
    </tr>
    <tr>
      <th>MiscFeature</th>
      <td>0.963014</td>
    </tr>
    <tr>
      <th>Alley</th>
      <td>0.937671</td>
    </tr>
    <tr>
      <th>Fence</th>
      <td>0.807534</td>
    </tr>
    <tr>
      <th>MasVnrType</th>
      <td>0.597260</td>
    </tr>
    <tr>
      <th>FireplaceQu</th>
      <td>0.472603</td>
    </tr>
    <tr>
      <th>LotFrontage</th>
      <td>0.177397</td>
    </tr>
    <tr>
      <th>GarageYrBlt</th>
      <td>0.055479</td>
    </tr>
    <tr>
      <th>GarageCond</th>
      <td>0.055479</td>
    </tr>
    <tr>
      <th>GarageType</th>
      <td>0.055479</td>
    </tr>
    <tr>
      <th>GarageFinish</th>
      <td>0.055479</td>
    </tr>
    <tr>
      <th>GarageQual</th>
      <td>0.055479</td>
    </tr>
    <tr>
      <th>BsmtFinType2</th>
      <td>0.026027</td>
    </tr>
    <tr>
      <th>BsmtExposure</th>
      <td>0.026027</td>
    </tr>
    <tr>
      <th>BsmtQual</th>
      <td>0.025342</td>
    </tr>
    <tr>
      <th>BsmtCond</th>
      <td>0.025342</td>
    </tr>
    <tr>
      <th>BsmtFinType1</th>
      <td>0.025342</td>
    </tr>
    <tr>
      <th>MasVnrArea</th>
      <td>0.005479</td>
    </tr>
    <tr>
      <th>Electrical</th>
      <td>0.000685</td>
    </tr>
    <tr>
      <th>Id</th>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>




```python
# We can eliminate columns with more than 10% empty values
eliminate = base.columns[(base.isnull().sum()/base.shape[0]) > 0.1]
eliminate
```




    Index(['LotFrontage', 'Alley', 'MasVnrType', 'FireplaceQu', 'PoolQC', 'Fence',
           'MiscFeature'],
          dtype='object')




```python
# Deleting these columns
base = base.drop(eliminate,axis=1)
```

- We want to create a first model to check how much we are getting wrong and then plan how to improve. To do this:
  - We will **eliminate the text columns**
  - We need to **handle empty values**
  - We will **choose some algorithms to test and an error evaluation method**


```python
# Selecting only numeric columns
col = base.columns[base.dtypes != 'object']
col
```




    Index(['Id', 'MSSubClass', 'LotArea', 'OverallQual', 'OverallCond',
           'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
           'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
           'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
           'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
           'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
           'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
           'MoSold', 'YrSold', 'SalePrice'],
          dtype='object')




```python
# And create a new base with these values
base2 = base.loc[:,col]
base2.head(3)
```





  <div id="df-acbe7963-a9a5-4188-b714-b99f387cde3b" class="colab-df-container">
    <div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>...</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>8450</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>196.0</td>
      <td>706</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>61</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>9600</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>0.0</td>
      <td>978</td>
      <td>0</td>
      <td>...</td>
      <td>298</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>11250</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>162.0</td>
      <td>486</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>223500</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 37 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-acbe7963-a9a5-4188-b714-b99f387cde3b')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">




  </div>


<div id="df-cd8264f6-e591-43d8-8581-2b823f2f5a73">
  <button class="colab-df-quickchart" onclick="quickchart('df-cd8264f6-e591-43d8-8581-2b823f2f5a73')"
            title="Suggest charts"
            style="display:none;">

  </button>


</div>

  </div>





```python
# Checking for empty values
base2.isnull().sum().sort_values(ascending=False).head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GarageYrBlt</th>
      <td>81</td>
    </tr>
    <tr>
      <th>MasVnrArea</th>
      <td>8</td>
    </tr>
    <tr>
      <th>Id</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>




```python
# Replacing empty values ​​with -1
base2 = base2.fillna(-1)
```

- This will be our initial base to start

## Creating our model

- **Let's separate it into training and testing**
    - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html


```python
# Selecting X and y
X = base2.drop('SalePrice',axis=1)
y = base2.SalePrice
```


```python
# Importing train_test_split
from sklearn.model_selection import train_test_split
```


```python
# Separating this base into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```

- **The next step is to select the algorithms we are going to use. We can start with the simplest algorithms such as:**
    - Linear Regression
        - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    - Regression Tree
        - https://scikit-learn.org/stable/modules/tree.html#regression
    - KNeighborsRegressor
        - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor


```python
# Importing linear regression
from sklearn.linear_model import LinearRegression
```


```python
# Creating the regressor and fit it with the training data
reg_rl = LinearRegression().fit(X_train, y_train)
```


```python
# Making the prediction for the test data
y_rl = reg_rl.predict(X_test)
```


```python
# Importing the regression tree
from sklearn import tree
```


```python
# Creating the regressor and fitting it with the training data
reg_ar = tree.DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
```


```python
# Making the prediction
y_ar = reg_ar.predict(X_test)
```


```python
# Importing KNN
from sklearn.neighbors import KNeighborsRegressor
```


```python
# Creating the regressor and fit it with the training data
reg_knn = KNeighborsRegressor(n_neighbors=2).fit(X_train, y_train)
```


```python
# Making the prediction
y_knn = reg_knn.predict(X_test)
```

- **And evaluate this data, using both the absolute and quadratic error:**
    - Mean absolute error
        - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
    - Mean squared error
        - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html


```python
# Importing the mean absolute error
from sklearn.metrics import mean_absolute_error
```


```python
# And the mean squared error
from sklearn.metrics import mean_squared_error
```


```python
# Evaluating regression error
print(mean_absolute_error(y_test,y_rl))
print(mean_squared_error(y_test,y_rl))
```

    23763.187393064567
    1533982883.444864
    


```python
# of the decision tree
print(mean_absolute_error(y_test,y_ar))
print(mean_squared_error(y_test,y_ar))
```

    27580.78838174274
    2530245114.701245
    


```python
# and knn
print(mean_absolute_error(y_test,y_knn))
print(mean_squared_error(y_test,y_knn))
```

    33273.08298755187
    2733937586.841286
    

- **We can visually plot the relationship of y_test to the predictions made**
    - For this we will use matplotlib
        - https://matplotlib.org/


```python
# Importing matplotlib
import matplotlib.pyplot as plt
```


```python
# Creating this graph
fig, ax = plt.subplots(ncols=3,figsize=(15,5))

ax[0].scatter(y_test/100000,y_rl/100000)
ax[0].plot([0,700000],[0,700000],'--r')
ax[1].scatter(y_test/100000,y_ar/100000)
ax[1].plot([0,700000],[0,700000],'--r')
ax[2].scatter(y_test/100000,y_knn/100000)
ax[2].plot([0,700000],[0,700000],'--r')

ax[0].set(xlim=(0, 7),ylim=(0, 7))
ax[0].set_xlabel('Real')
ax[0].set_ylabel('Prediction')
ax[1].set(xlim=(0, 7),ylim=(0, 7))
ax[1].set_xlabel('Real')
ax[1].set_ylabel('Prediction')
ax[2].set(xlim=(0, 7),ylim=(0, 7))
ax[2].set_xlabel('Real')
ax[2].set_ylabel('Prediction')

plt.show()
```


    

    


- **We will use Linear Regression because it was the algorithm with the lowest mean squared error, the same metric evaluated by Kaggle when classifying models.**

## Making the prediction for the competition test base


```python
# Importing the test base
test = pd.read_csv('test.csv')
```


```python
# Viewing the base
test.head(3)
```





  <div id="df-b913c4c4-ce34-4597-9d30-db740baceba3" class="colab-df-container">
    <div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1461</td>
      <td>20</td>
      <td>RH</td>
      <td>80.0</td>
      <td>11622</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>120</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1462</td>
      <td>20</td>
      <td>RL</td>
      <td>81.0</td>
      <td>14267</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Gar2</td>
      <td>12500</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1463</td>
      <td>60</td>
      <td>RL</td>
      <td>74.0</td>
      <td>13830</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 80 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-b913c4c4-ce34-4597-9d30-db740baceba3')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">



  </div>


<div id="df-8a13c301-f1e3-4294-a0f2-93ae62b4db2f">
  <button class="colab-df-quickchart" onclick="quickchart('df-8a13c301-f1e3-4294-a0f2-93ae62b4db2f')"
            title="Suggest charts"
            style="display:none;">

  </button>

</div>


  </div>




- **Now we will repeat the same treatments we did in the training base**
  - Note: **we cannot delete lines**


```python
# Deleting the same columns from the training base
test = test.drop(eliminate,axis=1)
```


```python
# Checking numeric columns
col2 = test.columns[test.dtypes != 'object']
col2
```




    Index(['Id', 'MSSubClass', 'LotArea', 'OverallQual', 'OverallCond',
           'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
           'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
           'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
           'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
           'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
           'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
           'MoSold', 'YrSold'],
          dtype='object')




```python
# Also keeping only the numeric columns
test = test.loc[:,col2]
```


```python
# Checking the remaining base
test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1459 entries, 0 to 1458
    Data columns (total 36 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Id             1459 non-null   int64  
     1   MSSubClass     1459 non-null   int64  
     2   LotArea        1459 non-null   int64  
     3   OverallQual    1459 non-null   int64  
     4   OverallCond    1459 non-null   int64  
     5   YearBuilt      1459 non-null   int64  
     6   YearRemodAdd   1459 non-null   int64  
     7   MasVnrArea     1444 non-null   float64
     8   BsmtFinSF1     1458 non-null   float64
     9   BsmtFinSF2     1458 non-null   float64
     10  BsmtUnfSF      1458 non-null   float64
     11  TotalBsmtSF    1458 non-null   float64
     12  1stFlrSF       1459 non-null   int64  
     13  2ndFlrSF       1459 non-null   int64  
     14  LowQualFinSF   1459 non-null   int64  
     15  GrLivArea      1459 non-null   int64  
     16  BsmtFullBath   1457 non-null   float64
     17  BsmtHalfBath   1457 non-null   float64
     18  FullBath       1459 non-null   int64  
     19  HalfBath       1459 non-null   int64  
     20  BedroomAbvGr   1459 non-null   int64  
     21  KitchenAbvGr   1459 non-null   int64  
     22  TotRmsAbvGrd   1459 non-null   int64  
     23  Fireplaces     1459 non-null   int64  
     24  GarageYrBlt    1381 non-null   float64
     25  GarageCars     1458 non-null   float64
     26  GarageArea     1458 non-null   float64
     27  WoodDeckSF     1459 non-null   int64  
     28  OpenPorchSF    1459 non-null   int64  
     29  EnclosedPorch  1459 non-null   int64  
     30  3SsnPorch      1459 non-null   int64  
     31  ScreenPorch    1459 non-null   int64  
     32  PoolArea       1459 non-null   int64  
     33  MiscVal        1459 non-null   int64  
     34  MoSold         1459 non-null   int64  
     35  YrSold         1459 non-null   int64  
    dtypes: float64(10), int64(26)
    memory usage: 410.5 KB
    


```python
# Viewing the number of empty values
test.isnull().sum().sort_values(ascending=False).head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GarageYrBlt</th>
      <td>78</td>
    </tr>
    <tr>
      <th>MasVnrArea</th>
      <td>15</td>
    </tr>
    <tr>
      <th>BsmtHalfBath</th>
      <td>2</td>
    </tr>
    <tr>
      <th>BsmtFullBath</th>
      <td>2</td>
    </tr>
    <tr>
      <th>BsmtUnfSF</th>
      <td>1</td>
    </tr>
    <tr>
      <th>GarageCars</th>
      <td>1</td>
    </tr>
    <tr>
      <th>GarageArea</th>
      <td>1</td>
    </tr>
    <tr>
      <th>BsmtFinSF1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>BsmtFinSF2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>TotalBsmtSF</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>



- **We will need to remove the empty values ​​because linear regression will not be able to work with empty values**
  - If we try to use this base, it will return an **error** saying that the **input has NaN values**
- We can just replace it with -1 as we did above


```python
# Replacing empty values ​​with -1
test = test.fillna(-1)
```

- **Now we can use our model and adjust the data to use in Kaggle**


```python
# Let's use linear regression to make the prediction
y_pred = reg_rl.predict(test)
```


```python
# We can add this forecast column to our database
test['SalePrice'] = y_pred
```


```python
# And extract only the Id and SalePrice
result = test[['Id','SalePrice']]
result.head(3)
```





  <div id="df-1207699a-a8a4-4a00-b356-1bd55605f358" class="colab-df-container">
    <div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1461</td>
      <td>122234.995960</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1462</td>
      <td>139178.263684</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1463</td>
      <td>169872.054251</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-1207699a-a8a4-4a00-b356-1bd55605f358')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">



  </div>


<div id="df-7dfbe493-7bda-4779-a6af-701e6db5fb53">
  <button class="colab-df-quickchart" onclick="quickchart('df-7dfbe493-7bda-4779-a6af-701e6db5fb53')"
            title="Suggest charts"
            style="display:none;">

  </button>

</div>


  </div>





```python
# We can then export this base
result.to_csv('result.csv',index=False)
```

- **Next steps**
  - We can try to improve the cleaning of your data
  - Then we can do feature engineering
  - Standardization / normalization of the data
  - And even variable selection
