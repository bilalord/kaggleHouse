# Submission Exercise

names = ['MSSubClass', 'LotArea', 'LandContour', 'Neighborhood', 'BldgType', 'HouseStyle',
                         'OverallQual', 'OverallCond', 'YearBuilt', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                         'MasVnrType', 'MasVnrArea', 'ExterQual', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1',
                         'HeatingQC', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'KitchenAbvGr', 'KitchenQual',
                         'Functional', 'Fireplaces', 'GarageCars', 'WoodDeckSF', 'OpenPorchSF', 'ScreenPorch',
                         'PoolArea', 'SaleCondition', "SalePrice"]

house_test = house_test[names]

from xgboost import XGBRegressor
import numpy as np
X_train = house_train.copy()
y_train = house_train.loc[:, "SalePrice"]

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
final_score = score_dataset(X_train, y_train, xgb)

print("Final RMSLE score:", final_score)

xgb = XGBRegressor(**xgb_params)
# XGB minimizes MSE, but competition loss is RMSLE
# So, we need to log-transform y to train and exp-transform the predictions
xgb.fit(X_train, np.log(y))
predictions = np.exp(xgb.predict(house_test))

output = pd.DataFrame({'Id': house_test.index, 'SalePrice': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")