import pandas as pd
import pickle

with open("../data/COMPLETE_df.pkl", 'rb') as f:
    x = pickle.load(f)

df = pd.DataFrame(x)

df_new = df.flux_list.apply(pd.Series).merge(df, right_index = True, left_index = True)
#     .drop(["ingredients"], axis = 1)

df_new.to_pickle('../data/complete-splited.pkl')