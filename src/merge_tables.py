import pickle
import pandas as pd
import time as time


def merge(from_sp, to_sp):
  t_start=time.time()

  df_spectra = pd.read_pickle('data/sdss/spectra_' + str(from_sp) + '-' + str(to_sp) + '.pkl')
  df_meta_data = pd.read_pickle('data/sdss/meta_table.pkl')


  # print('df_spectra.head \n', df_spectra)
  # print('df_meta_data \n', df_meta_data)

  df_meta_data["objid"] = df_meta_data['bestObjID'].astype(int)
  df_spectra['objid'] = df_spectra['objid'].astype(int)
  
  df_spectra.drop_duplicates('objid')
  df_meta_data.drop_duplicates()
  
  df_meta_data = df_meta_data.drop(columns={"specObjID"})

  df_merge = pd.merge(df_spectra, df_meta_data, on=['objid'])

  df_merge["dec"] = df_merge['dec_y']
  df_merge["z"] = df_merge['z_y']
  df_merge["ra"] = df_merge['ra_y']
  df_merge = df_merge.drop(columns={'dec_x', 'z_x', 'ra_x', 'dec_y', 'z_y', 'ra_y'})

  t_end=time.time()

  df_merge.to_pickle('data/sdss/spectra-meta-merged_' + str(from_sp) + '-' + str(to_sp) + '.pkl')

  t_delta = t_end - t_start
  print("time:", t_delta)


config = {
  'from_sp': 0,
  'to_sp': 5000,
  'run_merge': True
}

if config['run_merge'] == True:
  merge(config['from_sp'], config['to_sp'])

df = pd.read_pickle(
  'data/sdss/spectra-meta-merged_' + str(config['from_sp']) + '-' + str(config['to_sp']) + '.pkl'
)

