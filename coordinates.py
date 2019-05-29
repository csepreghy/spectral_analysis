import pandas as pd
import numpy as np
import astropy.coordinates as coord
import astropy.units as u
import time as time

start=time.time()

# Load the data
# Ra and Dec are now given in degrees
data = pd.read_csv("data/lowz.csv")
ra = data.iloc[0][0]
dec = data.iloc[0][1]

# Create a new dataframe, for the converted units of Ra and Dec
data_newunits = []

# Convert Ra to h,m,s and Dec to d,m,s and put it in the new dataframe
for i in range(len(data)):
    test_coords = coord.SkyCoord(ra=data.iloc[i][0], dec=data.iloc[i][1], unit=(u.degree, u.degree))
    data_newunits.append(str(test_coords.ra.to('hourangle')) + str(" +") + str(test_coords.dec.to('degree')))
    print(i)


print(data_newunits)

np.savetxt("coordinates_lowz.txt", data_newunits, fmt="%s")
end=time.time()

tt=end - start
print(tt)