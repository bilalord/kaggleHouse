import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def vis_cat (df, category):
    x = pd.DataFrame(df[category].value_counts())
    x = x.reset_index()
    ax = sns.barplot(x=category, y="index", data=x)
    ax.set(xlabel="Amount", ylabel="Categories")
    ax.set_title(category)
    return plt.show()