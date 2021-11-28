def clean(df):
    #Correct typos in object columns Exterior2nd, also checked all other object cols
    df["Exterior2nd"] = df["Exterior2nd"].replace({"Brk Cmn": "BrkComm"})
    df["Exterior2nd"] = df["Exterior2nd"].replace({"CemntBd": "CmentBd"})
    df["Exterior2nd"] = df["Exterior2nd"].replace({"WdShing": "Wd Shng"})

    # Some values of GarageYrBlt are corrupt, so we'll replace them
    # with the year the house was built
    df["GarageYrBlt"] = df["GarageYrBlt"].where(df.GarageYrBlt <= 2010, df.YearBuilt)

    # Names beginning with numbers are awkward to work with
    df.rename(columns={
        "1stFlrSF": "FirstFlrSF",
        "2ndFlrSF": "SecondFlrSF",
        "3SsnPorch": "Threeseasonporch",
    }, inplace=True,
    )
    return df