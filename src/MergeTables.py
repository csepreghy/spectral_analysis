import pickle
import pandas as pd
import time as time

start=time.time()

with open('../data/sdss/Part1-10-14.pkl', 'rb') as f:
    x = pickle.load(f)

df_dir=pd.DataFrame(x)
# print(x.values)
# df_dir.head(1)

with open('../data/sdss/direct_sql.pkl', 'rb') as f1:
    x1 = pickle.load(f1)

df=pd.DataFrame(x1)
# print(x.values)
# df.head()
# print(len(df_dir))

df_dir["plate"]=df["plate"]
df_dir["class"]=df["class"]
df_dir["zErr"]=df["zErr"]
df_dir["bestObjID"]= df["bestObjID"]


end=time.time()
df_dir.to_pickle('../data/sdss/FinalTable.pkl')

tt=end - start
print("time:", tt)