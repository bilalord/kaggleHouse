import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def impute_drop(df, cutoff):

# Numerical values treatment
# Find out what numerical columns have missing values
    list_num_null = []
    for name in df.select_dtypes("number"):
        if name == "SalePrice":
            continue
        if df[name].isnull().sum() > 0:
            list_num_null.append(name)
        else:
            continue
# Count how many missing values per category identified in the step before
    list_num_val = []
    for name in list_num_null:
        list_num_val.append(df[name].isnull().sum())
# Build descriptive DF about missing values
    df_desc = {"CatMissVal": list_num_null, "NbMissVal": list_num_val}
    df_desc = pd.DataFrame(data=df_desc)
    df_desc["PercMissVal"] = df_desc.NbMissVal.div(len(df.index))


# Categorical values treatment
# Find out which cat columns are missing data
    list_cat_null = []
    for name in df.select_dtypes("category"):
        if df[name].isnull().sum() > 0:
            list_cat_null.append(name)
        else:
            continue
# Count how many missing values per category identified in the step before
    list_cat_val = []
    for name in list_cat_null:
        list_cat_val.append(df[name].isnull().sum())
# Build DF with missing values
    df_desc_cat = {"CatMissVal" : list_cat_null, "NbMissVal" : list_cat_val}
    df_desc_cat = pd.DataFrame(data=df_desc_cat)
    df_desc_cat["PercMissVal"] = df_desc_cat.NbMissVal.div(len(df.index))

# Concat missing values in descriptive DF
    df_desc_complete = pd.concat([df_desc, df_desc_cat])

# Identify features with >30% missing data and adjust lists accordingly
    num_names = list(df_desc[(df_desc["PercMissVal"] > cutoff)].CatMissVal.unique())
    cat_names = list(df_desc_cat[(df_desc_cat["PercMissVal"] > cutoff)].CatMissVal.unique())
    list_num_null = list(filter(lambda x: x not in num_names, list_num_null))
    list_cat_null = list(filter(lambda x: x not in cat_names, list_cat_null))

# Drop cols in DF above cutoff
    df = df.drop(num_names, axis=1)
    df = df.drop(cat_names, axis=1)

# Imputation --------------------

# Impute every column missing values with median imputation
# By looking at boxplots/distplot value, we see median seems to be the best solution for num cat in general
    for cat in list_num_null:
        df[cat] = df[cat].fillna(df[cat].median())

# Impute categorical features with mode
    df[list_cat_null]=df[list_cat_null].fillna(df.mode().iloc[0])

# Finish by encoding categorical features with ordinalencoder

    all_cat_features = []
    for name in df.select_dtypes("category"):
        all_cat_features.append(name)

    enc = OrdinalEncoder()
    df[all_cat_features] = enc.fit_transform(df[all_cat_features])

# Output Report  ----------------

    print(len(num_names)," numerical features were dropped due to % missing data above", cutoff*100,":", num_names)
    print(len(cat_names)," categorical features were dropped due to % missing data above", cutoff*100,":", cat_names)

    return df, df_desc_complete