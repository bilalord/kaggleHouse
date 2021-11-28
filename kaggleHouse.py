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

# 1. Data preparation

# Clean data (typos, titles simplification)
from data_loading.clean import clean
df = clean(df)

# Encode data (Define each categorical value as such and add None for missing values)
from data_loading.encode import encode
df = encode(df)

# Impute data (fill empty cells) and drop cols with too many missing values arg: cutoff
from data_loading.impute_drop import impute_drop
df, df_impute_drop = impute_drop(df, 0.3)


