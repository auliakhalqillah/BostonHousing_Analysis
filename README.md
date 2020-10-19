# Boston Housing Analysis
## Introduction
In [Scikit-Learn](https://scikit-learn.org/stable/datasets/index.html) package of Python, there are some datasets that can be used to drill analysis. An example is Boston House Price. This dataset is used to predict the price of house in Boston through a regression model. I will show about analysis for this data to predict the price by using Python.

The boston data has 506 rows and 14 columns:
```
# - CRIM     per capita crime rate by town
# - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
# - INDUS    proportion of non-retail business acres per town
# - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# - NOX      nitric oxides concentration (parts per 10 million)
# - RM       average number of rooms per dwelling
# - AGE      proportion of owner-occupied units built prior to 1940
# - DIS      weighted distances to five Boston employment centres
# - RAD      index of accessibility to radial highways
# - TAX      full-value property-tax rate per $10,000
# - PTRATIO  pupil-teacher ratio by town
# - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# - LSTAT    % lower status of the population
# - MEDV     Median value of owner-occupied homes in $1000's
```

Our target is MEDV (median valu of owner-occupied homes in $1000's). By using another variabels, We can predict the price of house through the model process.
## Load Data
```
from sklearn import datasets
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data
boston = datasets.load_boston()
print('[1] Feature Description')
print(boston['DESCR'])
boston_data = boston['data']
feature_names = boston['feature_names']
# Make dataframe
df = pd.DataFrame(boston_data, columns=[feature_names])
# The boston['target'] is MEDV (Median value of owner-occupied). The name of feature is replaced by PRICE
df['PRICE'] = boston['target']
print('\n[2] Load Data')
print(df.head())
# Info data
print('\n[3] Info Data')
print(df.info())
```
The code above will show result
##### Result
```
[1] Feature Description
.. _boston_dataset:

Boston house prices dataset
---------------------------

**Data Set Characteristics:**  

    :Number of Instances: 506 

    :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.

    :Attribute Information (in order):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's

    :Missing Attribute Values: None

    :Creator: Harrison, D. and Rubinfeld, D.L.

This is a copy of UCI ML housing dataset.
https://archive.ics.uci.edu/ml/machine-learning-databases/housing/


This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.

The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
prices and the demand for clean air', J. Environ. Economics & Management,
vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
...', Wiley, 1980.   N.B. Various transformations are used in the table on
pages 244-261 of the latter.

The Boston house-price data has been used in many machine learning papers that address regression
problems.   
     
.. topic:: References

   - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
   - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.


[2] Load Data
      CRIM    ZN INDUS CHAS    NOX     RM   AGE     DIS  RAD    TAX PTRATIO       B LSTAT PRICE
0  0.00632  18.0  2.31  0.0  0.538  6.575  65.2  4.0900  1.0  296.0    15.3  396.90  4.98  24.0
1  0.02731   0.0  7.07  0.0  0.469  6.421  78.9  4.9671  2.0  242.0    17.8  396.90  9.14  21.6
2  0.02729   0.0  7.07  0.0  0.469  7.185  61.1  4.9671  2.0  242.0    17.8  392.83  4.03  34.7
3  0.03237   0.0  2.18  0.0  0.458  6.998  45.8  6.0622  3.0  222.0    18.7  394.63  2.94  33.4
4  0.06905   0.0  2.18  0.0  0.458  7.147  54.2  6.0622  3.0  222.0    18.7  396.90  5.33  36.2

[3] Info Data
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 506 entries, 0 to 505
Data columns (total 14 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   (CRIM,)     506 non-null    float64
 1   (ZN,)       506 non-null    float64
 2   (INDUS,)    506 non-null    float64
 3   (CHAS,)     506 non-null    float64
 4   (NOX,)      506 non-null    float64
 5   (RM,)       506 non-null    float64
 6   (AGE,)      506 non-null    float64
 7   (DIS,)      506 non-null    float64
 8   (RAD,)      506 non-null    float64
 9   (TAX,)      506 non-null    float64
 10  (PTRATIO,)  506 non-null    float64
 11  (B,)        506 non-null    float64
 12  (LSTAT,)    506 non-null    float64
 13  (PRICE,)    506 non-null    float64
dtypes: float64(14)
memory usage: 55.5 KB
None
```
Acording to info data, there are 506 rows data and 14 columns. The data doesn't have a missing value and all data types are float64. Then, We check the correlation for each pairwise of features
```
# How correlation for each pairwise of feature?
print('\n[4] Correlation')
correlation = df.corr()
print(correlation)
fig1 = sns.heatmap(correlation, cmap='seismic')
fig1.set_title('Correlation for each pairwise of features')
fig1.set_ylabel(None)
```
It will show a figure as follows

![fig1](https://github.com/auliakhalqillah/BostonHousing_Analysis/blob/main/Boston_Corr_Pairwaise_1.png)

Based on these correlation, We take MEDV's column and sorted it.
```
# Sort the PRICE coefficient correaltion
print('\n[5] Sort the PRICE coefficient correaltion')
corr_medv_value = correlation.iloc[0:13,13].to_numpy()
corr_index_name = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
corr_medv_df = pd.DataFrame(corr_medv_value, columns=['CORR_MEDV'], index=corr_index_name).sort_values(by='CORR_MEDV')
print(corr_medv_df)

corr_medv_df.plot(kind='bar')
plt.xlabel('Features')
plt.ylabel('Coefficient Correlation')
plt.title('The correlation coefficient for each feature to PRICE (Sorted)')
```
![fig2](https://github.com/auliakhalqillah/BostonHousing_Analysis/blob/main/Boston_Corr_of_Price_2.png)

We take two higher correlation features for each sign, there are LSTAT and PTRATIO for negative correlation, ZN and RM for positive correlation. The LSTAT is the higher correlation for negative correlation and the PTRATIO is following behind. The negative correlation indicate these features are inversely correlated to PRICE. That means the higher LSTAT and PTRATIO, the lower PRICE. However, the LSTAT is more significantly affect to PRICE than PTRATIO. On the other hand, The RM is the higher correlation for positive correlation, and the ZN is following behind. The positive correlation indicates these features are directly propotional correlated to PRICE. That means, the higher RM and ZN, the higher PRICE. However, the RM is more significantly affect to PRICE. By this information, the CHAS is not much significantly affect to PRICE. It means, the presence or absence of tract bound river is not much affect to median value of occupied homes.

```
# Short Describe
print('\n[6] Short Describe')
print(df.describe())
```

The minimum of PRICE is $5.000 and maximum of PRICE is $50.0000. The mean of PRICE $22.532 with standard deviation of $9.197.

Next, We create a model to predict a price of house. Model prediction is created by using Linear Regression. The data is splitted to training data (75 %) and testing data (25 %). We set the X variabel is all features excepts MEDV and y is a target that is MEDV. Note, for this code, variable name of MEDV is replaced to PRICE.

```
X = df[['LSTAT','RM']]
y = df[['PRICE']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
linreg = LinearRegression()
linreg = linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
```

The y_pred is predicted price from testing data. Then, We combine the actual price (y_true) and prdicted price (y_pred) to single dataframe.

```
# Make dataframe
lin_model = pd.DataFrame(y_pred, columns=['Predicted_MEDV'])
lin_model['Actual_MEDV'] = y_test.to_numpy()
print('\n[7] Model')
print(lin_model.head(10))
```
##### Result
```
[7] Model
   Predicted_MEDV  Actual_MEDV
0       26.222386         22.6
1       24.158101         50.0
2       24.291621         23.0
3       12.900551          8.3
4       22.342989         21.2
5       22.956271         19.9
6       21.302473         20.6
7       23.075558         18.7
8       15.823157         16.1
9       24.364505         18.6
```

Next, We try to plot the correlation between actual price with predicted price through linear regression (order = 1)

```
# linear model
sns.lmplot('Predicted_MEDV', 'Actual_MEDV',lin_model, line_kws={'color':'black'})
fig4 = plt.gca()
fig4.set_title('Linear Regression Model')
fig4.set_xlabel('Median Value ($1000)')
fig4.set_ylabel('Median Value ($1000)')
```

![fig3](https://github.com/auliakhalqillah/BostonHousing_Analysis/blob/main/Boston_First_Orde_Reg_Mod_3.png)

As We can see, the black line is not fit to the data yet. Becasue the black line is linear  So, We try to set the order equal to 2.

```
# Second order model
sns.lmplot('Predicted_MEDV', 'Actual_MEDV',lin_model, order=2, line_kws={'color':'black'})
fig5 = plt.gca()
fig5.set_title('Second Order Regression Model')
fig5.set_xlabel('Median Value ($1000)')
fig5.set_ylabel('Median Value ($1000)')
```

![fig4](https://github.com/auliakhalqillah/BostonHousing_Analysis/blob/main/Boston_Sec_Orde_Reg_Mod_4.png)

As We can see, the black line is more fit to the data than before. It means, the data model is a second order equation. Finally, We try to evaluate this model

 ```
 print('\n[8] Evaluation model')
print('MSE:', mean_squared_error(y_test, y_pred, squared=True))
print('RMSE:', mean_squared_error(y_test, y_pred, squared=False))
print('R2:', r2_score(y_test, y_pred))
```
##### Result
```
[8] Evaluation model
MSE: 35.19229684718286
RMSE: 5.9323095710846765
R2: 0.5692445415835348
```
Based on evaluation model, by 35.19 of MSE and 5.932 of RMSE, this model is good enough to prredict a median value of occupied home and by 56.92 % of variance, the observed data can be explained by using this model.

** NOTE: Mean Square Error indicates the quality of model. The small MSE (close to zero), then the model is good. R2 score indicates proportion of the variance of dependent variable that's explained by an independent variable or variables in a regression model. If the R2 of a model is 0.50, then approximately half of the observed variation can be explained by the model's inputs.**

```
# Model Distribution of PRICE
plt.figure(6)
plt.hist(y_pred, bins=10, edgecolor='black')
plt.xlabel('Median Value ($1000)')
plt.ylabel('count')
plt.title('Model Distribution of PRICE')
```
![fig5](https://github.com/auliakhalqillah/BostonHousing_Analysis/blob/main/Boston_Model_Dist_Price_5.png)

Based on distribution of price model data, the price of occupied home is in range of $17.000 - $25.000.

We test this model to predict a price of house. # Predict Assume, We have two clients who want to predict the home prices. The first client has 17% lower status of the population and need 5 rooms. The second client has 20% lower status of the population and need 3 rooms. By these illustrations, the second client has high peverty from the first client.

```
data = [[17, 5],[20, 3]]
predict_price = linreg.predict(data)
print('\n[9] Predict Prices')
for i, price in enumerate(predict_price):
    print ("Predicted selling price for Client %d : $%f" % (i+1, price*1000))
```
###### Result
```
[9] Predict Prices
Predicted selling price for Client 1 : $13017.737143
Predicted selling price for Client 2 : $958.474824
```

## Conclusions
1. The higher percentage lower status of the population (LSTAT) affects to the cheap price of home (PRICE)
2. The higher number of rooms per dwellling affects to the high price of home (PRICE)
3. By 56.92 % of variance, the observed data can be explained by using this model. That means, the price of home can be predicted by using this model thorugh LSTAT and RM parameters, but with moderate precision.

