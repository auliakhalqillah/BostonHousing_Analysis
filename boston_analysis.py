# Objective : Predict the median value of occupied homes.
# Source    : sklearn datasets
# Date      : 2020, October 16th

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
# Acording to info data, there are 506 rows data and 14 columns. 
# The data doesn't have a missing value and all data types are float64.

# How correlation for each pairwise of feature?
print('\n[4] Correlation')
correlation = df.corr()
print(correlation)
fig1 = sns.heatmap(correlation, cmap='seismic')
fig1.set_title('Correlation for each pairwise of features')
fig1.set_ylabel(None)

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

# We take two higher correlation features for each sign, there are LSTAT and PTRATIO for negative correlation, ZN and RM for 
# positive correlation. The LSTAT is the higher correlation for negative correlation and the PTRATIO is following behind.
# The negative correlation indicate these features are inversely correlated to PRICE. That means the higher LSTAT and PTRATIO,
# the lower PRICE. However, the LSTAT is more significantly affect to PRICE than PTRATIO. On the other hand, The RM is the higher
# correlation for positive correlation, and the ZN is following behind. The positive correlation indicates these features
# are directly propotional correlated to PRICE. That means, the higher RM and ZN, the higher PRICE. However, the RM is more
# significantly affect to PRICE. By this information, the CHAS is not much significantly affect to PRICE. It means, the presence
# or absence of tract bound river is not much affect to median value of occupied homes. 

# Short Describe
print('\n[6] Short Describe')
print(df.describe())
# The minimum of PRICE is $5.000 and maximum of PRICE is $50.0000. The mean of PRICE 22.532806 with standard deviation of 9.197104.

# MODEL-PREDICT
# Because the PTRATIO and ZN are not significantly affect to PRICE, We only use LSTAT and RM to generate a model. 
# This model will be used to predict PRICE as a target. The data is splitted to training data (75%) and testing data (25%).
X = df[['LSTAT','RM']]
y = df[['PRICE']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
linreg = LinearRegression()
linreg = linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)

# Make dataframe
lin_model = pd.DataFrame(y_pred, columns=['Predicted_MEDV'])
lin_model['Actual_MEDV'] = y_test.to_numpy()
print('\n[7] Model')
print(lin_model.head(10))
# linear model
sns.lmplot('Predicted_MEDV', 'Actual_MEDV',lin_model, line_kws={'color':'black'})
fig4 = plt.gca()
fig4.set_title('Linear Regression Model')
fig4.set_xlabel('Median Value ($1000)')
fig4.set_ylabel('Median Value ($1000)')
# Second order model
sns.lmplot('Predicted_MEDV', 'Actual_MEDV',lin_model, order=2, line_kws={'color':'black'})
fig5 = plt.gca()
fig5.set_title('Second Order Regression Model')
fig5.set_xlabel('Median Value ($1000)')
fig5.set_ylabel('Median Value ($1000)')

# Evaluation Model
# Mean Square Error indicates the quality of model. The small MSE (close to zero), then the model is good.
# R2 score indicates proportion of the variance of dependent variable that's explained by an independent 
# variable or variables in a regression model. If the R2 of a model is 0.50, then approximately half of 
# the observed variation can be explained by the model's inputs.

print('\n[8] Evaluation model')
print('MSE:', mean_squared_error(y_test, y_pred, squared=True))
print('RMSE:', mean_squared_error(y_test, y_pred, squared=False))
print('R2:', r2_score(y_test, y_pred))

# The first order of linear regression is not suitable for this model but the second order of linear regression instead.
# By 35.19 of MSE and 5.932 of RMSE, this model is good enough to prredict a median value of occupied home and by 56.92 %
# of variance, the observed data can be explained by using this model.

# Model Distribution of PRICE
plt.figure(6)
plt.hist(y_pred, bins=10, edgecolor='black')
plt.xlabel('Median Value ($1000)')
plt.ylabel('count')
plt.title('Model Distribution of PRICE')
# Based on model data, the price of occupied home is in range of $17.000 - $25.000.

# Predict
# Assume, We have two clients who want to predict the home prices.
# The first client has 17% lower status of the population and need 5 rooms.
# The second client has 20% lower status of the population and need 3 rooms.
# By these illustrations, the second client has high peverty from the first client.
data = [[17, 5],[20, 3]]
predict_price = linreg.predict(data)
print('\n[9] Predict Prices')
for i, price in enumerate(predict_price):
    print ("Predicted selling price for Client %d : $%f" % (i+1, price*1000))

# Conclusions
# 1. The higher percentage lower status of the population (LSTAT) affects to the cheap price of home (PRICE)
# 2. The higher number of rooms per dwellling affects to the high price of home (PRICE)
# 3. By 56.92 % of variance, the observed data can be explained by using this model. That means, the price of
# home can be predicted by using this model thorugh LSTAT and RM parameters.

plt.show()