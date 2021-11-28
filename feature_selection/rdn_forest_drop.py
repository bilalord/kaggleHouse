# Uses random forest to find importance of features in large dataset df. According to the importance value (float type), drops columns below or equal to it and scores
# the remaining features with the RMSLE. The function returns a plot of the relative importance of each feature (if set to True), the amount of features dropped and the list
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def rdn_forest_drop (df, importance_value, bool_plot):

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
    indices = np.argsort(importances)[-70:]  # top 10 features
    
    if bool_plot == True:
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()

    #Drop columns below treshold
    imp_df = pd.DataFrame({"Feature": features, "Importance": importances})
    drop_rdn_for = imp_df["Feature"].iloc[imp_df.loc[imp_df["Importance"] < importance_value].index.tolist()]
    df = df.drop(columns=drop_rdn_for, axis=1)
    print(len(drop_rdn_for), " features dropped by RandomForest:", list(drop_rdn_for))

    return df
