# Variation of rdn_forest_drop but with exploratory purposes. Will score the dataset after removing features according to an relative importance.
# The function will loop over each importance level until importance_limit given by the user. In the end, plots the result in a line plot for importance lvl vs RMSLE score
# Returns result dataframe as well

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
