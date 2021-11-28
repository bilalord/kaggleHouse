from pandas.api.types import CategoricalDtype
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor


#Plot styling
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=14, titlepad=10)


# Import data set
house_train = pd.read_csv("Kagglehouse/train_house.csv")
house_test = pd.read_csv("Kagglehouse/test_house.csv")

# Merge to treat together/consistency
df = pd.concat([house_train, house_test])

# -------------Data Pretreatment-------------

# 1. Data cleaning (typos, titles simplification)
def clean(df):
    #Correct typos in object columns Exterior2nd, also checked all other object cols
    df["Exterior2nd"] = df["Exterior2nd"].replace({"Brk Cmn": "BrkComm"})
    df["Exterior2nd"] = df["Exterior2nd"].replace({"CemntBd": "CmentBd"})
    df["Exterior2nd"] = df["Exterior2nd"].replace({"WdShing": "Wd Shng"})

    # Some values of GarageYrBlt are corrupt, so we'll replace them
    # with the year the house was built
    df["GarageYrBlt"] = df["GarageYrBlt"].where(df.GarageYrBlt <= 2010, df.YearBuilt)

    # Names beginning with numbers are awkward to work with
    df.rename(columns={
        "1stFlrSF": "FirstFlrSF",
        "2ndFlrSF": "SecondFlrSF",
        "3SsnPorch": "Threeseasonporch",
    }, inplace=True,
    )
    return df

df = clean(df)

# 2. Encode data (Define each categorical value as such and add None for missing values)
def encode(df):
    # The numeric features are already encoded correctly (`float` for
    # continuous, `int` for discrete), but the categoricals we'll need to
    # do ourselves. Note in particular, that the `MSSubClass` feature is
    # read as an `int` type, but is actually a (nominative) categorical.

    # The nominative (unordered) categorical features
    features_nom = ["MSSubClass", "MSZoning", "Street", "Alley", "LandContour", "LotConfig", "Neighborhood",
                    "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st",
                    "Exterior2nd", "MasVnrType", "Foundation", "Heating", "CentralAir", "GarageType", "MiscFeature",
                    "SaleType", "SaleCondition"]

    # The ordinal (ordered) categorical features

    # Pandas calls the categories "levels"
    five_levels = ["Po", "Fa", "TA", "Gd", "Ex"]
    ten_levels = list(range(10))

    ordered_levels = {
        "OverallQual": ten_levels,
        "OverallCond": ten_levels,
        "ExterQual": five_levels,
        "ExterCond": five_levels,
        "BsmtQual": five_levels,
        "BsmtCond": five_levels,
        "HeatingQC": five_levels,
        "KitchenQual": five_levels,
        "FireplaceQu": five_levels,
        "GarageQual": five_levels,
        "GarageCond": five_levels,
        "PoolQC": five_levels,
        "LotShape": ["Reg", "IR1", "IR2", "IR3"],
        "LandSlope": ["Sev", "Mod", "Gtl"],
        "BsmtExposure": ["No", "Mn", "Av", "Gd"],
        "BsmtFinType1": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
        "BsmtFinType2": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
        "Functional": ["Sal", "Sev", "Maj1", "Maj2", "Mod", "Min2", "Min1", "Typ"],
        "GarageFinish": ["Unf", "RFn", "Fin"],
        "PavedDrive": ["N", "P", "Y"],
        "Utilities": ["NoSeWa", "NoSewr", "AllPub"],
        "CentralAir": ["N", "Y"],
        "Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
        "Fence": ["MnWw", "GdWo", "MnPrv", "GdPrv"],
    }

    # Add a None level for missing values
    ordered_levels = {key: ["None"] + value for key, value in
                      ordered_levels.items()}

    # Nominal categories
    for name in features_nom:
        df[name] = df[name].astype("category")
        # Add a None category for missing values
        if "None" not in df[name].cat.categories:
            df[name].cat.add_categories("None", inplace=True)
    # Ordinal categories
    for name, levels in ordered_levels.items():
        df[name] = df[name].astype(CategoricalDtype(levels, ordered=True))
    return df

df = encode(df)

# 3. Impute data (fill empty cells)

# Find out what numerical columns have missing values
"""list_num_null = []
for name in df.select_dtypes("number"):
    if df[name].isnull().sum() > 0:
        list_num_null.append(name)
    if df[name] == "SalePrice":
        continue
    else:
        continue"""
# Count how many missing values per category identified in the step before
"""list_num_val = []
for name in list_num_null:
    list_num_val.append(df[name].isnull().sum())"""

# Build DF with descriptives values about missing values for numerical data
list_num_null = ["LotFrontage","MasVnrArea",'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath',
                 'BsmtHalfBath','GarageCars','GarageArea']
list_num_val = [486, 23, 1, 1, 1, 1, 2, 2, 1, 1]

df_desc = {"CatMissVal" : list_num_null, "NbMissVal" : list_num_val}
df_desc = pd.DataFrame(data=df_desc)
df_desc["PercMissVal"] = df_desc.NbMissVal.div(len(df.index))

# Visualize data to decide how to impute
"""sns.boxplot(df.LotFrontage)"""
"""sns.distplot(df.LotFrontage)"""

# Impute every column missing values with median imputation
# By looking at boxplots value, we see median seems to be the best solution for num cat in general
for cat in list_num_null:
    df[cat] = df[cat].fillna(df[cat].median())

# Find out what categorical columns have missing values
"""list_cat_null = []
for name in df.select_dtypes("category"):
    if df[name].isnull().sum() > 0:
        list_cat_null.append(name)
    else:
        continue"""

# Build DF with descriptives values about missing values for categorical data
list_cat_null = ['MSZoning', 'Alley', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'BsmtQual', 'BsmtCond',
 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType',
 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType']
list_cat_val = [4, 2721, 2, 1, 1, 24, 81, 82, 82, 79, 80, 1, 1, 2, 1420, 157, 159, 159, 159, 2909, 2348, 2814, 1]

df_desc_cat = {"CatMissValCat" : list_cat_null, "NbMissVal" : list_cat_val}
df_desc_cat = pd.DataFrame(data=df_desc_cat)
df_desc_cat["PercMissVal"] = df_desc_cat.NbMissValCat.div(len(df.index))

# Features Alley, PoolQC, Fence and MiscFeature have >93% missing values, drop columns from df
# After further considerations, it is advised to drop >30% missing values variable, drop FireplaceQu from df
df = df.drop(["Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature"], axis=1)
df_desc_cat = df_desc_cat.drop([1,14,19,20,21], axis=0)


# Impute remaining columns
# Impute missing data before encoding categorical columns
# Visualize data to decide how to impute
"""sns.boxplot(df.LotFrontage)"""
"""sns.distplot(df.LotFrontage)"""
# Fill cat columns with most frequent, except FireplaceQu and SalePrice, here is the list (same as before minus Fire
# and SalePrice.
list_miss_cat = ['MSZoning', 'Utilities', 'OverallQual', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'BsmtQual',
                 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'KitchenQual', 'Functional',
                 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'SaleType']

# Visualize Categorical data (category is STR type)
def vis_cat (df, category):
    x = pd.DataFrame(df[category].value_counts())
    x = x.reset_index()
    ax = sns.barplot(x=category, y="index", data=x)
    ax.set(xlabel="Amount", ylabel="Categories")
    ax.set_title(category)
    plt.show()

df[list_miss_cat]=df[list_miss_cat].fillna(df.mode().iloc[0])

# SalePrice last column with missing values
# 4. Encode all categorical data with OrdinalEncoder

all_cat_features = []
for name in df.select_dtypes("category"):
    all_cat_features.append(name)

enc = OrdinalEncoder()
df[all_cat_features] = enc.fit_transform(df[all_cat_features])


# 4. Resplit data after pretreatment is done
house_train = df.iloc[:1460]
house_test = df.iloc[:1460]

# -------------Data Pretreatment OVER-------------
# -------------Data Visualization-----------------
# -------------Features Selection-----------------

def score_dataset(X, y, model=XGBRegressor()):
    # Label encoding for categoricals
    #
    # Label encoding is good for XGBoost and RandomForest, but one-hot
    # would be better for models like Lasso or Ridge. The `cat.codes`
    # attribute holds the category levels.
    for colname in X.select_dtypes(["category"]):
        X[colname] = X[colname].cat.codes
    # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
    log_y = np.log(y)
    score = cross_val_score(
        model, X, log_y, cv=5, scoring="neg_mean_squared_error",
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score

# Determine INITIAL performance with score_dataset  // INITIAL Baseline score: 0.14653 RMSLE
"""X = house_train.copy()
y = X.pop("SalePrice")

baseline_score = score_dataset(X, y)
print(f"Baseline score: {baseline_score:.5f} RMSLE")"""

# RandomForestRegressor to determine feature importance VS SalePrice (from p. 164 book Approaching ML)
# With this spliced dataset: Baseline score: 0.16652 RMSLE, worse than with all features! Deleted process

# Check amount of variation in each column and remove what is below X% (enter relative value)
# Outputs DF and prints nb features dropped and which

def var_drop (df, rel_value):
    copy_df = df.copy()
    var_train = copy_df.var()
    mean_train = copy_df.mean()
    CV = var_train/mean_train
    var_df = pd.DataFrame({"Mean": mean_train, "StDev": var_train, "Coefficient":CV})
    var_df["<10%"] = var_df["Coefficient"] < rel_value
    drop_var_features = var_df.index[var_df["<10%"]==True].tolist()
    df_out = copy_df.drop(drop_var_features, axis=1)
    print(len(drop_var_features), " Features dropped: ", drop_var_features)

    return df_out

# Variance analysis
# With following parameters:
house_train_novar = var_drop(house_train, 0.01)
# 2  Features dropped:  ['Street', 'YrSold']
X = house_train_novar.copy()
y = X.pop("SalePrice")
baseline_score = score_dataset(X, y)
print(f"Baseline score: {baseline_score:.5f} RMSLE")
# Baseline score: 0.14132 RMSLE


# Build correlation table to check if there are strongly related features that can be removed
# Calculated with Pearson's coefficients
"""house_corr = house_train_novar.corr()"""
"""sns.heatmap(data=house_corr)"""
# Removing the >0.85 correlation columns (GarageYrBlt and GarageCars) did not improve RMSLE
# RMSLE: 0.14649

# Check low correlation with SalePrice, removing features with correlation up to 40% does not improve RMSLE
# Same remark with high correlation features
def corr_drop (df, min_corr):

    #Build dataframe with the correlation analyzed
    corr_values = np.arange(0.01,min_corr,0.01)
    corr_output_df = pd.DataFrame(columns=["Correlation %","# Dropped", "RMSLE Score"], index=range(len(corr_values)))
    corr_output_df["Correlation %"] = corr_values

    #Enter how many categories are dropped in output dataframe
    drop_cat_list = []
    for value in range(len(corr_values)):
        drop_cat = abs(df.corr()["SalePrice"][abs(df.corr()\
        ["SalePrice"]) > corr_values[value]].drop("SalePrice")).index.tolist()
        drop_cat_list.append(len(drop_cat))
    corr_output_df["# Dropped"] = drop_cat_list

    #Enter RMSLE scores in dataframe
    scores = []
    reset_df = df.copy()
    for i in range(len(corr_values)):
        df = reset_df.copy()
        drop_cat_score = abs(df.corr()["SalePrice"][abs(df.corr()\
        ["SalePrice"]) > corr_values[i]].drop("SalePrice")).index.tolist()
        if len(drop_cat_score) == 0:
            X = df.copy()
            y = X.pop("SalePrice")
            baseline_score = score_dataset(X, y)
            scores.append(baseline_score)
        else:
            df = df.drop(drop_cat_score, axis=1)
            X = df.copy()
            y = X.pop("SalePrice")
            baseline_score = score_dataset(X, y)
            scores.append(baseline_score)
    corr_output_df["RMSLE Score"] = scores

    g = sns.PairGrid(corr_output_df, y_vars=["# Dropped"], x_vars=["Correlation %", "RMSLE Score"])

    return g.map(sns.scatterplot)

# Exploratory Rdn forest to find best importance value
# Best is at 0.0003
def rdn_forest_look (df, importance_limit):

    A = df.copy()
    y = A.pop("SalePrice")

    if "Id" in A.columns:
        A = A.drop("Id", axis=1)
    if "SalePrice" in A.columns:
        A = A.drop("SalePrice", axis=1)

    model = RandomForestRegressor(random_state=1, max_depth=10)
    model.fit(A,y)
    features = A.columns
    importances = model.feature_importances_
    indices = np.argsort(importances)[-73:]  # top 73features

    imp_range = np.arange(0,importance_limit,0.0001)
    rdn_for_df = pd.DataFrame(columns=["Importance value", "RMSLE Score"], index=range(len(imp_range)))
    imp_ = pd.DataFrame({"Feature": features, "Importance": importances})
    RMSLE_list = []
    for i in imp_range:
        E = A.copy()
        #Drop columns below treshold
        imp_df = imp_.copy()
        drop_rdn_for = imp_df["Feature"].iloc[imp_df.loc[imp_df["Importance"] <= i].index.tolist()]
        E = E.drop(columns=drop_rdn_for, axis=1)
        score = score_dataset(E,y)
        RMSLE_list.append(score)

    rdn_for_df["Importance value"] = imp_range
    rdn_for_df["RMSLE Score"] = RMSLE_list

    sns.lineplot(x="Importance value", y="RMSLE Score", data=rdn_for_df)

    return rdn_for_df

# Random forest check function, with desired importance value
def rdn_forest_drop (df, importance_value, bool_plot):

    A = df.copy()
    y = A.pop("SalePrice")

    if "Id" in A.columns:
        A = A.drop("Id", axis=1)
    if "SalePrice" in A.columns:
        A = A.drop("SalePrice", axis=1)

    model = RandomForestRegressor(random_state=1, max_depth=10)
    model.fit(A,y)
    features = A.columns
    importances = model.feature_importances_
    indices = np.argsort(importances)[-73:]  # top 73features

    if bool_plot == True:
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()

    #Drop columns below treshold
    imp_df = pd.DataFrame({"Feature": features, "Importance": importances})
    drop_rdn_for = imp_df["Feature"].iloc[imp_df.loc[imp_df["Importance"] <= importance_value].index.tolist()]
    A = A.drop(columns=drop_rdn_for, axis=1)
    score = score_dataset(A,y)
    print(len(drop_rdn_for), " features dropped. RMSLE Score is: ", score)
    print(drop_rdn_for)

    return A

house_train_novar_nofor = rdn_forest_drop(house_train_novar, 0.0003, False)
# 10 Features dropped
#  6            Utilities
#  28            BsmtCond
#  32        BsmtFinType2
#  36             Heating
#  39          Electrical
#  42        LowQualFinSF
#  59          GarageQual
#  60          GarageCond
#  65    Threeseasonporch
#  68             MiscVal
# New RMSLE Score: 0.1409131405422512



# https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/




