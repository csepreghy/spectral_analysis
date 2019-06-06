
import pandas as pd

import time as time

from src.import_data import get_save_SDSS_from_coordinates
from src.SDSS_direct_query import query
from src.MergeTables import merge
from src.neural_network_classifier import run_neural_network

# df = pd.read_hdf('train.h5')
# model = run_neural_network(df, batch_size=5, hidden_layers=[128, 128], n_epochs=5)

# gama_df = get_gama_dataset_from_csv()
# sp, xid = get_sample_SDSS_dataset_from_query()


# coord_list = ['0h8m05.63s +14d50m23.3s', '0h9m06.63s +15d55m23.3s']
# with open('data/sdss_coordinates_lowz.txt') as text_file:
#   coord_list = text_file.read().splitlines()
#   # mystring.replace('\n', ' ').replace('\r', '')

query()

coord_list=pd.read_csv("data/lowz.csv")
# coord_list.head()
start=time.time()
ra_list = coord_list["ra"].tolist()
dec_list= coord_list["dec"].tolist()
end=time.time()


tt=end - start
print("time for listing is:", tt)



start1=time.time()
ra=ra_list[15001:30000]
dec=dec_list[15001:30000]
get_save_SDSS_from_coordinates( ra , dec )
end1=time.time()

tt1= end1- start1
length=len(ra) -1
print("time for "+str(length)+" stellar objects:" , tt1)

merge(length)

#test