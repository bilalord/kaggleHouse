import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import KFold, cross_val_score

# Import data set
house_train = pd.read_csv("Kagglehouse/train_house.csv")
house_test = pd.read_csv("Kagglehouse/test_house.csv")

# Merge to treat together/consistency
df = pd.concat([house_train, house_test])

from kaggleHouse