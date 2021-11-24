#Drops features with low coefficient of variation. Chose cutoff value with rel_value (in absolute numbers).
#Produces a dataframe with the statistical description of the dataset

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
