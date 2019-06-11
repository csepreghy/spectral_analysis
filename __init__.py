import pandas as pd
import time as time

from src.import_data import get_save_SDSS_from_coordinates
from src.SDSS_direct_query import query
from src.merge_tables import merge

from src.data_preprocessing import filter_sources, spectrum_cutoff, create_continuum, merge_lines_and_continuum
from src.get_spectrallines import get_spectrallines

# with open('data/sdss_coordinates_lowz.txt') as text_file:
#   coord_list = text_file.read().splitlines()
#     mystring.replace('\n', ' ').replace('\r', '')

# query()

# coord_list=pd.read_csv("data/lowz.csv")
# start=time.time()
# ra_list = coord_list["ra"].tolist()
# dec_list= coord_list["dec"].tolist()
# end=time.time()
# tt=end - start
# print("time for listing is:", tt)

# start1=time.time()
# ra=ra_list[40001:45000]
# dec=dec_list[40001:45000]
# get_save_SDSS_from_coordinates( ra , dec )
# end1=time.time()

# tt1= end1- start1
# length=len(ra) -1
# print("time for "+str(length)+" stellar objects:" , tt1)

# merge(length)


#spectra = pd.read_pickle('data/alldatamerged.pkl')

#df_continuum = pd.read_pickle('continuum_df.pkl')
df_spectral_lines = pd.read_pickle('spectral_lines_df.pkl')


start = time.time()


spectra = pd.read_pickle('data/alldatamerged.pkl')
df_filtered = filter_sources(df = spectra)
print('DF Filtered: ')
print(df_filtered.head())
"""
df_spectral_lines = get_spectrallines(df_filtered)
print('Spectral Lines')
print(df_spectral_lines.head())
df_spectral_lines.to_pickle('spectral_lines2_df.pkl')
"""
df_cutoff = spectrum_cutoff(df = df_filtered)
df_continuum = create_continuum(df = df_cutoff, sigma=16, downsize=8)
df_continuum.to_pickle('continuum_df.pkl')


df_complete = merge_lines_and_continuum(df_spectral_lines, df_continuum)
print("DF Complete: ")
print(df_complete.head())
df_complete.to_pickle('COMPLETE_df.pkl')

end = time.time()
tt = end - start
print(" ")
print("Time elapsed: ", tt, "s")
print(tt/60, "min")

# model = create_model(df_final, configs['neural_network'])


