import pandas as pd
import numpy as np
import pickle
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import time as time
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score as acc
from sklearn.preprocessing import LabelEncoder as le
from sklearn.model_selection import train_test_split

with open("../data/COMPLETE_df.pkl", 'rb') as f:

    x = pickle.load(f)
df = pd.DataFrame(x)

df_new=df.flux_list.apply(pd.Series) \
    .merge(df, right_index = True, left_index = True)
#     .drop(["ingredients"], axis = 1)

df_new.to_pickle('../data/complete-splited.pkl')