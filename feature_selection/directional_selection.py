import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression
import numpy as np
from feature_selection.score_dataset import score_dataset
import seaborn as sns

def directional_selection(df,nb_features, forward_bool, assess_bool):
    if assess_bool == True:
        if forward_bool == True:
            len_df = df.columns.value_counts().sum()-1
            range_feat = np.arange(1, nb_features+1, 1).tolist()

        if forward_bool == False:
            len_df = df.columns.value_counts().sum()-1
            range_feat = np.arange(len_df, nb_features-1, -1).tolist()

        estimator = LinearRegression()
        dir_sel_df = pd.DataFrame(columns=["Features left", "RMSLE Score"], index=range(len(range_feat)))
        dir_sel_df["Features left"] = range_feat
        scores = []

        for i in range_feat:
            A = df.copy()
            sfs1 = sfs(estimator, k_features=i, forward=forward_bool, scoring='neg_mean_squared_error')
            X = df.copy()
            y = X.pop("SalePrice")
            sfs1 = sfs1.fit(X, y)
            feat_names = list(sfs1.k_feature_names_)
            feat_names.append("SalePrice")
            A = df[feat_names]
            X = A.copy()
            y = X.pop("SalePrice")
            scoring = score_dataset(X,y)
            scores.append(scoring)
        dir_sel_df["RMSLE Score"] = scores

        sns.lineplot(x="Features left", y="RMSLE Score", data=dir_sel_df)

        return dir_sel_df

    if assess_bool == False:
        estimator = LinearRegression()
        sfs1 = sfs(estimator, k_features=nb_features, forward=forward_bool, scoring='neg_mean_squared_error')
        X = df.copy()
        y = X.pop("SalePrice")
        sfs1 = sfs1.fit(X, y)
        feat_names = list(sfs1.k_feature_names_)
        print("Features chosen by directional selection:", list(feat_names))
        feat_names.append("SalePrice")
        A = df[feat_names]
        X = A.copy()
        y = X.pop("SalePrice")
        scoring = score_dataset(X, y)

        print("RMSLE Score after directional selection:", round(scoring, 5))

        return A