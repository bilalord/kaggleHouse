import pandas as pd
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

plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

#Import data
def load_data():
    #Import data set
    house_train = pd.read_csv("train_house.csv")
    house_test = pd.read_csv("test_house.csv")

    #Merge to treat together/consistency
    df = pd.concat([house_train, house_test])

    #Preprocessing (clean, encode, impute)
    df = clean(df)
    df = encode(df)
    df = impute(df)

    # Reform splits
    house_train = df.loc[house_train.index, :]
    house_test = df.loc[house_test.index, :]
    return house_train, house_test

#Data cleaning (typos, titles simplification)
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

#Define each categorical value as such and add None for missing values
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

#Impute 0 for missing numerical values and None for missing categorical values
def impute(df):
    for name in df.select_dtypes("number"):
        df[name] = df[name].fillna(0)
    for name in df.select_dtypes("category"):
        df[name] = df[name].fillna("None")
    return df

def plot_variance(pca, width=8, dpi=100):
    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )
    # Set up figure
    fig.set(figwidth=8, dpi=100)
    return axs

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


house_train, house_test = load_data()

#Numerical features list creation
feature_types = house_train.dtypes
features = house_train.columns.unique()
num_features = []

for i in range(len(feature_types)):
    if feature_types[i] == "int64" or feature_types[i] == "float64":
        num_features.append(features[i])
    else:
        continue


X = house_train.copy()
X = X.loc[:, num_features]
y = X.pop('SalePrice')

# Standardize
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
# Create principal components
pca = PCA()
X_pca = pca.fit_transform(X)

# Convert to dataframe
component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=component_names)

X_pca.drop(columns=X_pca.columns[0],
        axis=1,
        inplace=True)
X_pca.drop(columns=X_pca.columns[0],
        axis=1,
        inplace=True)
X_pca.drop(columns=X_pca.columns[0],
        axis=1,
        inplace=True)


plot_variance(pca)
mi_scores = make_mi_scores(X_pca, y, discrete_features=False)
loadings = pd.DataFrame(
    pca.components_.T,  # transpose the matrix of loadings
    columns=component_names,  # so the columns are the principal components
    index=X.columns,  # and the rows are the original features
)

print(X_pca.head())
print(mi_scores)
print(loadings)
plt.show()


