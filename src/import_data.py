from astroquery.gama import GAMA
import pandas as pd
import numpy as np

def get_gama_dataset_from_query():
  gama_all_data = GAMA.query_sql('SELECT * FROM AATSpecAll AS s WHERE s.z BETWEEN 0.000000001 AND 10')
  gama_all_data_df = pd.DataFrame(np.array(gama_all_data)) # convert table to CSV
  gama_all_data_df.to_csv('data/GAMA.csv') # write csv into data folder

  return gama_all_data_df


def get_gama_dataset_from_csv():
  gama_df = pd.read_csv('data/GAMA.csv')
  
  return gama_df
