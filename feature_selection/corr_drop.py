# Checks for low correlation (Pearson's) features with the target category (here SalePrice) according to cutoff "min_corr" chosen by the user. min_corr is entered relative to 1.
# The function proceeds to score (RMSLE) the dataset with each subsequent feature drop and returns 2 plots showing the change in RMSLE with the % of correlation analyze and with
# the amount of features dropped

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


#lol