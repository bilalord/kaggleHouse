import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def vis_cat (not_imputed_df, imputed_df, category):
    fig, ax = plt.subplots(1, 2)
    sns.countplot(not_imputed_df[category], data=not_imputed_df, ax=ax[0]).set(title="Before Imputation")
    sns.countplot(imputed_df[category], data=imputed_df, ax=ax[1]).set(title="After Imputation")
    return fig.show()





