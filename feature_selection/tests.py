import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Observe data
cat_nam = []
for name in house_train.select_dtypes("int64"):
    cat_nam.append(name)
for nb, name in enumerate(cat_nam):
    plo = nb+1
    plt.subplot(3, 6, plo)
    sns.scatterplot(x=house_train[name], y=house_train["SalePrice"], data=house_train)
plt.show()

#Data normalization tests + scoring
for i in range(len(A["LotArea"])):
    if A["LotArea"].iloc[i] > 50000:
        A["LotArea"].iloc[i] = A["LotArea"].median()
    else:
        continue
for i in range(len(A["WoodDeckSF"])):
    if A["WoodDeckSF"].iloc[i] == 0:
        A["WoodDeckSF"].iloc[i] = A["WoodDeckSF"].mean()
    else:
        continue
X = A.copy()
y = X.pop("SalePrice")
from feature_selection.score_dataset import score_dataset
score_dataset(X,y)

#Regressor test
from xgboost import XGBRegressor
X_train = df_train.copy()
y_train = df_train.loc[:, "SalePrice"]

xgb_params = dict(
    max_depth=6,           # maximum depth of each tree - try 2 to 10
    learning_rate=0.01,    # effect of each tree - try 0.0001 to 0.1
    n_estimators=1000,     # number of trees (that is, boosting rounds) - try 1000 to 8000
    min_child_weight=1,    # minimum number of houses in a leaf - try 1 to 10
    colsample_bytree=0.7,  # fraction of features (columns) per tree - try 0.2 to 1.0
    subsample=0.7,         # fraction of instances (rows) per tree - try 0.2 to 1.0
    reg_alpha=0.5,         # L1 regularization (like LASSO) - try 0.0 to 10.0
    reg_lambda=1.0,        # L2 regularization (like Ridge) - try 0.0 to 10.0
    num_parallel_tree=1,   # set > 1 for boosted random forests
)

xgb = XGBRegressor(**xgb_params)
score_dataset(X_train, y_train, xgb)
