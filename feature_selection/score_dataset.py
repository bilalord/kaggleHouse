# Not mine, from KAGGLE.COM!
# https://www.kaggle.com/ryanholbrook/feature-engineering-for-house-prices
# RYAN HOLBROOK & ALEXIS COOK
from xgboost import XGBRegressor
import numpy as np
from sklearn.model_selection import KFold, cross_val_score


def score_dataset(X, y, model=XGBRegressor()):
    # Label encoding for categoricals
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
