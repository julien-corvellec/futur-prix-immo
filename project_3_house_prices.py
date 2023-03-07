"""Project 4 - House Prices - Advanced Regression Techniques.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv('train.csv')

"""## Exploratory Data Analysis"""

# numerical columns
df_train.select_dtypes(include=['int64', 'float64']).columns

len(df_train.select_dtypes(include=['int64', 'float64']).columns)

# categorical columns
df_train.select_dtypes(include=['object']).columns

len(df_train.select_dtypes(include=['object']).columns)

"""## Dealing with null values"""

# check if there are any null values
df_train.isnull().values.any()

# check how many nullvalues in the dataset
df_train.isnull().values.sum()

# check the number of null values in each column
df_train.isnull().sum()

# the list of columns which has null values
df_train.columns[df_train.isnull().any()]

# number of columns with null values
len(df_train.columns[df_train.isnull().any()])

# show the null values using the heatmap
plt.figure(figsize=(16,9))
sns.heatmap(df_train.isnull())
plt.show()

# get the percentage of null values
null_percent = df_train.isnull().sum() / df_train.shape[0] * 100

# number of missing values in each column / total values in that column

null_percent

# select the columns which has the null values more than 50%
col_for_drop = null_percent[null_percent > 50].keys()

col_for_drop
# null values are more than 50%

df_train = df_train.drop(labels=col_for_drop, axis=1)
# drop those columns which has more than 50% null values

df_train.shape
# 4 columns dropped

# the list of columns which has null values
df_train.columns[df_train.isnull().any()]

# number of columns with null values
len(df_train.columns[df_train.isnull().any()])

"""Add column mean in numerical columns"""

df_train['LotFrontage'] = df_train['LotFrontage'].fillna(df_train['LotFrontage'].mean())
df_train['MasVnrArea'] = df_train['MasVnrArea'].fillna(df_train['MasVnrArea'].mean())
df_train['GarageYrBlt'] = df_train['GarageYrBlt'].fillna(df_train['GarageYrBlt'].mean())

# number of columns with null values
len(df_train.columns[df_train.isnull().any()])

# the list of columns which has null values
df_train.columns[df_train.isnull().any()]

df_train['MasVnrType'] = df_train['MasVnrType'].fillna(df_train['MasVnrType'].mode()[0])
df_train['BsmtQual'] = df_train['BsmtQual'].fillna(df_train['BsmtQual'].mode()[0])
df_train['BsmtCond'] = df_train['BsmtCond'].fillna(df_train['BsmtCond'].mode()[0])
df_train['BsmtExposure'] = df_train['BsmtExposure'].fillna(df_train['BsmtExposure'].mode()[0])
df_train['BsmtFinType1'] = df_train['BsmtFinType1'].fillna(df_train['BsmtFinType1'].mode()[0])
df_train['BsmtFinType2'] = df_train['BsmtFinType2'].fillna(df_train['BsmtFinType2'].mode()[0])
df_train['Electrical'] = df_train['Electrical'].fillna(df_train['Electrical'].mode()[0])
df_train['FireplaceQu'] = df_train['FireplaceQu'].fillna(df_train['FireplaceQu'].mode()[0])
df_train['GarageType'] = df_train['GarageType'].fillna(df_train['GarageType'].mode()[0])
df_train['GarageFinish'] = df_train['GarageFinish'].fillna(df_train['GarageFinish'].mode()[0])
df_train['GarageQual'] = df_train['GarageQual'].fillna(df_train['GarageQual'].mode()[0])
df_train['GarageCond'] = df_train['GarageCond'].fillna(df_train['GarageCond'].mode()[0])

# number of columns with null values
len(df_train.columns[df_train.isnull().any()])

# check if there are any null values
df_train.isnull().values.any()

# check how many nullvalues in the dataset
df_train.isnull().values.sum()

df_train.head()

# show the null values using the heatmap
plt.figure(figsize=(16,9))
sns.heatmap(df_train.isnull(), cmap='coolwarm')
plt.show()

"""## Distplot"""

# describe the target
df_train['SalePrice'].describe()

# plot the distplot of target value
plt.figure(figsize=(16,9))
bar = sns.distplot(df_train['SalePrice'])
bar.legend(["Skewness: {:.2f}".format(df_train['SalePrice'].skew())]) # skewness
plt.show()
# Skewness is a measure of the asymmetry of the probability distribution of a real-valued random variable about its mean
# The skewness value can be positive, zero, negative, or undefined

"""## Correlation matrix and Heatmap"""

df_train_2 = df_train.drop(columns='SalePrice')

df_train_2.corrwith(df_train['SalePrice']).plot.bar(
    figsize=(16,9), title = 'Correlation with SalePrice', 
    rot = 45, grid = True
)

# correlation heatmap
plt.figure(figsize=(25,25))
ax = sns.heatmap(df_train.corr(), cmap='coolwarm', annot=True, linewidths=2)

# correlation heatmap of highly correlated features with 'SalePrice'
high_corr = df_train.corr()

high_corr_features = high_corr.index[abs(high_corr['SalePrice']) >= 0.5]

high_corr_features

len(high_corr_features)

# correlation heatmap of highly correlated features with 'SalePrice'
plt.figure(figsize=(16,9))
ax = sns.heatmap(df_train[high_corr_features].corr(), cmap='coolwarm', annot=True, linewidths=2)

"""# Part 2: Data preprocessing (test.csv)"""

df_test = pd.read_csv('/content/test.csv')

# columns with numerical variables
df_test.select_dtypes(include=['int64', 'float64']).columns

len(df_test.select_dtypes(include=['int64', 'float64']).columns)

# columns with categorical variables
df_test.select_dtypes(include=['object']).columns

len(df_test.select_dtypes(include=['object']).columns)

"""## Dealing with null values"""

# check if there are any null values
df_test.isnull().values.any()

# check how many nullvalues in the dataset
df_test.isnull().values.sum()

# check the number of null values in each column
df_test.isnull().sum()

# the list of columns which has null values
df_test.columns[df_test.isnull().any()]

# number of columns with null values
len(df_test.columns[df_test.isnull().any()])

# show the null values using the heatmap
plt.figure(figsize=(16,9))
sns.heatmap(df_test.isnull())
plt.show()

df_test.shape

# get the percentage of null values
null_percent = df_test.isnull().sum() / df_test.shape[0] * 100

# number of missing values in each column / total values in that column

null_percent

# select the columns which has the null values more than 50%
col_for_drop = null_percent[null_percent > 50].keys()

col_for_drop
# null values are more than 50%

df_test = df_test.drop(labels=col_for_drop, axis=1)
# drop those columns which has more than 50% null values

df_test.shape
# 5 columns dropped

# the list of columns which has null values
df_test.columns[df_test.isnull().any()]

# number of columns with null values
len(df_test.columns[df_test.isnull().any()])

"""Add column mean in numerical columns"""

df_test['LotFrontage'] = df_test['LotFrontage'].fillna(df_test['LotFrontage'].mean())
df_test['MasVnrArea'] = df_test['MasVnrArea'].fillna(df_test['MasVnrArea'].mean())
df_test['BsmtFinSF1'] = df_test['BsmtFinSF1'].fillna(df_test['BsmtFinSF1'].mean())
df_test['BsmtFinSF2'] = df_test['BsmtFinSF2'].fillna(df_test['BsmtFinSF2'].mean())
df_test['BsmtUnfSF'] = df_test['BsmtUnfSF'].fillna(df_test['BsmtUnfSF'].mean())
df_test['BsmtFullBath'] = df_test['BsmtFullBath'].fillna(df_test['BsmtFullBath'].mean())
df_test['BsmtHalfBath'] = df_test['BsmtHalfBath'].fillna(df_test['BsmtHalfBath'].mean())
df_test['TotalBsmtSF'] = df_test['TotalBsmtSF'].fillna(df_test['TotalBsmtSF'].mean())
df_test['GarageCars'] = df_test['GarageCars'].fillna(df_test['GarageCars'].mean())
df_test['GarageArea'] = df_test['GarageArea'].fillna(df_test['GarageArea'].mean())
df_test['GarageYrBlt'] = df_test['GarageYrBlt'].fillna(df_test['GarageYrBlt'].mean())

# number of columns with null values
len(df_test.columns[df_test.isnull().any()])

"""Add column mode in categorical columns"""

df_test['MSZoning'] = df_test['MSZoning'].fillna(df_test['MSZoning'].mode()[0])
df_test['Utilities'] = df_test['Utilities'].fillna(df_test['Utilities'].mode()[0])
df_test['Exterior1st'] = df_test['Exterior1st'].fillna(df_test['Exterior1st'].mode()[0])
df_test['Exterior2nd'] = df_test['Exterior2nd'].fillna(df_test['Exterior2nd'].mode()[0])
df_test['MasVnrType'] = df_test['MasVnrType'].fillna(df_test['MasVnrType'].mode()[0])

df_test['BsmtQual'] = df_test['BsmtQual'].fillna(df_test['BsmtQual'].mode()[0])
df_test['BsmtCond'] = df_test['BsmtCond'].fillna(df_test['BsmtCond'].mode()[0])
df_test['BsmtExposure'] = df_test['BsmtExposure'].fillna(df_test['BsmtExposure'].mode()[0])
df_test['BsmtFinType1'] = df_test['BsmtFinType1'].fillna(df_test['BsmtFinType1'].mode()[0])
df_test['BsmtFinType2'] = df_test['BsmtFinType2'].fillna(df_test['BsmtFinType2'].mode()[0])

df_test['KitchenQual'] = df_test['KitchenQual'].fillna(df_test['KitchenQual'].mode()[0])
df_test['Functional'] = df_test['Functional'].fillna(df_test['Functional'].mode()[0])
df_test['GarageType'] = df_test['GarageType'].fillna(df_test['GarageType'].mode()[0])
df_test['GarageFinish'] = df_test['GarageFinish'].fillna(df_test['GarageFinish'].mode()[0])
df_test['GarageQual'] = df_test['GarageQual'].fillna(df_test['GarageQual'].mode()[0])

df_test['GarageCond'] = df_test['GarageCond'].fillna(df_test['GarageCond'].mode()[0])
df_test['SaleType'] = df_test['SaleType'].fillna(df_test['SaleType'].mode()[0])

# number of columns with null values
len(df_test.columns[df_test.isnull().any()])

# check if there are any null values
df_test.isnull().values.any()

# check how many nullvalues in the dataset
df_test.isnull().values.sum()

df_test.head()

# show the null values using the heatmap
plt.figure(figsize=(16,9))
sns.heatmap(df_test.isnull(), cmap='coolwarm')
plt.show()

"""# Part 3: Handling the categorical features"""

df_train.shape

df_test.shape

# Copy train (excluding target) and test data 

df_train_cp = df_train.drop(['SalePrice', 'Id'], axis=1).copy()
df_test_cp = df_test.drop('Id', axis=1).copy()

df_train_cp.shape

df_test_cp.shape

# Combine train and test for One Hot Encoding
combined_data = pd.concat([df_train_cp, df_test_cp])

combined_data.shape

# Do One Hot Encoding for categorical features
combined_data = pd.get_dummies(combined_data, drop_first=True)

combined_data.shape

# Separate Train data and test data
x_train = combined_data.iloc[:1460, :]
x_test = combined_data.iloc[1460:, :]
y_train = df_train["SalePrice"]

x_train.shape

x_test.shape

y_train.shape

"""# Part 4: Training the model

## XGBoost
"""

from xgboost import XGBRegressor

regressor = XGBRegressor()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

y_pred

# read the submission file
sub_f = pd.read_csv('/content/sample_submission.csv')
sub_f.shape

sub_f = sub_f['Id']
type(sub_f)

# create a dictionary
d = {'Id':sub_f,
     'SalePrice':y_pred}

# combine two series object in a dataframe
df = pd.DataFrame(d)

df.head()

# save the file
df.to_csv('submission_new.csv', index=False)

"""# Part 5: Hyper Parameter Optimization"""

booster=['gbtree','gblinear']
learning_rate=[0.05,0.1,0.15,0.20]
max_depth = [2, 3, 5, 10, 15]
min_child_weight=[1,2,3,4]
n_estimators = [100, 500, 900, 1100, 1500]
base_score=[0.25,0.5,0.75,1]

# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'booster':booster,
    'learning_rate':learning_rate,
    'max_depth':max_depth,
    'min_child_weight':min_child_weight,
    'n_estimators': n_estimators,
    'base_score':base_score
    }

from sklearn.model_selection import RandomizedSearchCV

# random search with 4-fold cross validation
random_cv = RandomizedSearchCV(estimator=regressor,
            param_distributions=hyperparameter_grid,
            cv=5, n_iter=50,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5, 
            return_train_score = True,
            random_state=42)

random_cv.fit(x_train,y_train)

random_cv.best_estimator_

random_cv.best_params_

"""# Part 6: Final Model (XGBoost)"""

# train the model again with tunned hyperparameters
regressor = XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.1, max_delta_step=0,
             max_depth=2, min_child_weight=1, missing=None, n_estimators=900,
             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=1, verbosity=1)
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
y_pred

# read the submission file
sub_f = pd.read_csv('/content/sample_submission.csv')
sub_f.shape

sub_f = sub_f['Id']
type(sub_f)

# create a dictionary
d = {'Id':sub_f,
     'SalePrice':y_pred}

# combine two series object in a dataframe
df = pd.DataFrame(d)
df.head()

# save the file
df.to_csv('submission.csv', index=False)

