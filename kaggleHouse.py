import pandas as pd

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

# Remove ID feature
house_train = house_train.drop(columns="Id", axis=1)

# Check the variation in each column and remove what is below X% (enter relative value)
# Outputs DF, CV analysis report DF and prints nb features dropped and which
# Scores dataset after variation splicing
from feature_selection.var_drop import var_drop
house_train, cv_var_df = var_drop(house_train,0.01)
X = house_train.copy()
y = X.pop("SalePrice")
score_cv = score_dataset(X,y)
print("RMSLE Score after CV splicing:", round(score_cv,5))

# Check for Pearson correlation versus target "SalePrice"
# Functions outputs correlation dataframe with #features dropped for each cutoff% up to threshold set
# Also outputs scatter plots with scoring for each correlation% tested
# After testing, did not show improvement in RMSLE score by removing high or low correlated features
# Optional step, purely exploratory
# Removing the >0.85 correlation columns (GarageYrBlt and GarageCars) did not improve RMSLE
# Check low correlation with SalePrice, removing features with correlation up to 40% does not improve RMSLE
from feature_selection.corr_drop import corr_drop
"""correlation_df = corr_drop(house_train,0.3)"""

# Random Forest to check for importance versus "SalePrice" with cutoff importance level
# Data exploration made using rdn_forest_look to check for different importance scores the scoring effect
# Selected importance value with lowest RMSLE score on a wide range
# Outputs lineplot of RMSLE score versus importance value
from feature_selection.rdn_forest_look import rdn_forest_look
"""rdn_forest_look(house_train,0.01)"""
# Outputs spliced DF with or without importance plot according to bool
# Score dataset after random forest sorting
from feature_selection.rdn_forest_drop import rdn_forest_drop
house_train = rdn_forest_drop(house_train, 0.0003, False)
X = house_train.copy()
y = X.pop("SalePrice")
rdn_forest_score = score_dataset(X,y)
print("RMSLE Score after RDNForest splicing:", round(rdn_forest_score,5))

# Function directional_selection executes directional feature elimination using neg_mean_squared_error as
# scoring method (bc of the possible negative values, took the square instead of log)
# with a linear fit. Function outputs results DF and displays lineplot of results with seaborn
# Backward selection did not improve RMSLE from 60 down to 20 features (using forward_bool = False)
# Forward selection improved RMSLE with 32 features (using forward_bool = True) down to a score of 0.13880
from feature_selection.directional_selection import directional_selection
"""dir_selection_output = directional_selection(house_train, 40, True)"""
# Therefore, run forward feature selection with features_nb = 32 and score the new dataset
# All that can be done with same function as for exploration by setting assess_bool to False
B = directional_selection(house_train, 32, True, False)





