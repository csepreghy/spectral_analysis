from astroquery.sdss import SDSS
import pandas as pd
import time as time

# lowz=pd.read_csv("../data/lowz.csv")
# length=len(lowz)
#
# print(length)
start=time.time()

query = "select top 100000  \
z, ra, dec, bestObjID, plate , class, zErr  \
from specObj \
where 1.0 >z   and \
z >0 and \
zWarning = 0"
res = SDSS.query_sql(query)

df=res.to_pandas()
# df.head()

end=time.time()
df.to_pickle('../data/sdss/direct_sql.pkl')
tt=end - start
print("time consuming:", tt)
