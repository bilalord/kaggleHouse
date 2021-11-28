def impute(df):
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

# Impute every column missing values with median imputation
# By looking at boxplots/distplot value, we see median seems to be the best solution for num cat in general
    for cat in list_num_null:
        df[cat] = df[cat].fillna(df[cat].median())

###   ------------------------------------------------------------------------------------------------
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

# Identify features with >30% missing data


# Concat missing values DFs
df_desc = pd.concat([df_desc, df_desc_cat])