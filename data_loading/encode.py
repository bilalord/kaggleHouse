from pandas.api.types import CategoricalDtype

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