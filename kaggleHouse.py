import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

# Import data set
house_train = pd.read_csv("train_house.csv")
house_test = pd.read_csv("test_house.csv")

# Merge to treat together/consistency
df = pd.concat([house_train, house_test])

# --------1. Data preparation/visualization--------

# Clean data (typos, titles simplification)
from data_loading.clean import clean
df = clean(df)

# Encode data (Define each categorical value as such and add None for missing values)
from data_loading.encode import encode
df = encode(df)

# Impute data (fill empty cells) and drop cols with too many missing values arg: cutoff
# Also encoding categorical data with ordinal encoder
from data_loading.impute_drop import impute_drop
df_imputed, df_impute_drop = impute_drop(df, 0.3)

# Can visualize categorical features distribution before/after imputation with vis_cat
"""from data_loading.vis_cat import vis_cat
vis_cat(df, df_imputed, "GarageType")"""

# Resplit data to enter feature selection process
house_train = df_imputed.iloc[:1460]
house_test = df_imputed.iloc[:1460]

# --------2. Feature selection--------

# Score train dataset before any selection, establishing baseline score
from feature_selection.score_dataset import  score_dataset
X = house_train.copy()
y = X.pop("SalePrice")
baseline = score_dataset(X,y)
print("RMSLE Score before feature selection:", round(baseline,5))

# Check the variation in each column and remove what is below X% (enter relative value)
# Outputs DF, CV analysis report DF and prints nb features dropped and which
# Scores dataset after variation splicing
from feature_selection.var_drop import var_drop
house_train, cv_var_df = var_drop(house_train,0.01)
X = house_train.copy()
y = X.pop("SalePrice")
baseline = score_dataset(X,y)
print("RMSLE Score after variation splicing:", round(baseline,5))

# Check for Pearson correlation versus target "SalePrice"
# Functions outputs correlation dataframe with #features dropped for each cutoff% up to threshold set
# Also outputs scatter plots with scoring for each correlation% tested
# After testing, did not show improvement in RMSLE score by removing high or low correlated features
# Optional step, purely exploratory
# Removing the >0.85 correlation columns (GarageYrBlt and GarageCars) did not improve RMSLE
# Check low correlation with SalePrice, removing features with correlation up to 40% does not improve RMSLE
from feature_selection.corr_drop import corr_drop
"""correlation_df = corr_drop(house_train,0.3)"""





