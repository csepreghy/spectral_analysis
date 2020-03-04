import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, filters, feature
import sys
from bisect import bisect_left
import time as time
from tqdm.auto import tqdm

# -------------------------------
# Functions

def apply_gaussian_filter(fluxes, sigma):
  return filters.gaussian(image=fluxes, sigma=sigma)


def closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """

    pos = bisect_left(myList, myNumber)
    if pos == 0: return myList[0]
    if pos == len(myList): return myList[-1]
    
    before = myList[pos - 1]
    after = myList[pos]
    
    if after - myNumber < myNumber - before: return after
    else: return before


def find_midpoint(flux_interval, wavelength_interval, gradient_interval, spectralline_wavelength):
	"""
	Function that returns the closest critical point to the spectral line. If none is found, it returns a NaN.
	"""
	criticalpoints = []
	for i in range(len(flux_interval)-1):
		# Check if the derivative changes sign between i and i+1
		if (gradient_interval[i] < 0) != (gradient_interval[i + 1] < 0):
			criticalpoints.append(wavelength_interval[i])
	if len(criticalpoints) == 0:
		return np.nan
	elif len(criticalpoints) == 1:
		return criticalpoints[0]

	else:
		return closest(criticalpoints, spectralline_wavelength)



def find_criticalpoint(flux_interval, wavelength_interval, gradient_interval):
	"""
	Function that returns the first critical point it finds in the interval. If none is found, it returns a NaN.
	"""
	for i in range(len(flux_interval)-1):
		# Check if the derivative changes sign between i and i+1
		if (gradient_interval[i] < 0) != (gradient_interval[i + 1] < 0):
			wavelength_critical = wavelength_interval[i]
			return wavelength_critical
	
	return np.nan


def find_criticalpoint_extended(flux_interval, wavelength_interval, gradient_interval):
	"""
	Function that returns the first critical point it finds in the interval. If none is found, it returns a NaN.
	It has a delayed start, which is useful for finding the MgII spectral line.
	"""
	for i in range(30, len(flux_interval)-1):
		# Check if the derivative changes sign between i and i+1
		if (gradient_interval[i] < 0) != (gradient_interval[i + 1] < 0):
			wavelength_critical = wavelength_interval[i]
			return wavelength_critical

	return np.nan


# -------------------------------

# Useful website: http://www.stat.cmu.edu/tr/tr828/tr828.pdf

# The spectral lines
qso_lines = ['MgII_em', 'Hgamma_em', 'Hbeta_em', 'OIII_em', 'OIII_em', 'Halpha_em', 'OII_em']
star_lines = ['Halpha_ab', 'Hbeta_ab', 'Hgamma_ab', 'Hdelta_ab', 'NaI_ab']
gal_lines = ['Na_ab', 'Mg_ab', 'Halpha_em', 'S2_em', 'Hbeta_em', 'Gband_ab', 'CAIIH_ab', 'CAIIK_ab', 'OII_em']#, 'balmer_break']

wl_qso = [2799, 4342, 4861, 4960, 5008, 6565, 3727]
wl_star = [6565, 4861, 4340, 4101, 5896]
wl_gal = [5893, 5175, 6565, 6716, 4861, 4304, 3933.7, 3968, 3727] #, 4000]

# Complete spectral lines
speclines_name = ['MgII_em', 'Hgamma_em', 'Hbeta_em', 'OIII_em', 'OIII_em', 'Halpha_em', 'OII_em','Hdelta_ab',
                  'NaI_ab', 'Mg_ab', 'S2_em', 'Gband_ab', 'CAIIH_ab', 'CAIIK_ab']
speclines = [2799, 4342, 4861, 4960, 5008, 6565, 3727, 4101, 5895, 5175, 6716, 4304, 3933.7, 3968]

# Sort lists on wavelengths
sort_index = np.argsort(speclines)
speclines = np.array(speclines)[sort_index]
speclines_name = np.array(speclines_name)[sort_index]


def continuum(X, slope, intercept):
	return slope * X + intercept

def spectrallines_1source(flux, wavelength, z, sigma=4, delta1=10, delta2=80):
    """
    :param flux: Array of flux values of 1 source.
    :param wavelength: Array of wavelength values of 1 source.
    :param z: Redshift of the source.
    :param sigma: Smoothing parameter (default is 4).
    :param delta1: Interval in which to look for the exact midpoint of the peak (default is 5).
    :param delta2: Interval in which to look for the begin and end points of the peak (default is 80).
    :return: Vector with Pseudo-Equivalent Widths (EW) for each spectral line (for 1 source).
    """

    # Smooth the flux and compute its gradient
    smoothflux = apply_gaussian_filter(flux, sigma=sigma)

    gradient = np.gradient(smoothflux, wavelength)

    # The spectral lines EW will be saved in this list
    final_vector = []

    for s in range(len(speclines)):
        # -------- Step 1: find the exact midpoint of spectral peak --------

        # Look for the critical points within an interval of delta around the predicted peaks.
        line_min = speclines[s] * (1 + z) - delta1
        line_max = speclines[s] * (1 + z) + delta1
        interval_flux = smoothflux[(line_min < wavelength) & (wavelength < line_max)]
        interval_wavelength = np.array(wavelength[(line_min < wavelength) & (wavelength < line_max)])
        interval_gradient = gradient[(line_min < wavelength) & (wavelength < line_max)]

        # If the spectral line is outside of the wavelength range: EW = 0
        if len(interval_flux) == 0.0:
            EW = 0.0
            final_vector.append(EW)
            continue

        # Find the exact midpoint in a small interval
        wavelength_mid = find_midpoint(interval_flux, interval_wavelength, interval_gradient, speclines[s] * (1 + z))

        # If still no critical point is found: use location of spectral line
        if np.isnan(wavelength_mid):
            wavelength_mid = speclines[s] * (1+z)


        # -------- Step 2: find the begin and end points --------

        # Define the intervals to look at
        end_right = wavelength_mid - delta2
        end_left = wavelength_mid + delta2
        interval_r_flux = np.flip(smoothflux[(end_right < wavelength) & (wavelength < wavelength_mid)])
        interval_r_wavelength = np.flip(np.array(wavelength[(end_right < wavelength) & (wavelength < wavelength_mid)]))
        interval_r_gradient = np.flip(gradient[(end_right < wavelength) & (wavelength < wavelength_mid)])
        interval_l_flux = smoothflux[(wavelength_mid < wavelength) & (wavelength < end_left)]
        interval_l_wavelength = np.array(wavelength[(wavelength_mid < wavelength) & (wavelength < end_left)])
        interval_l_gradient = gradient[(wavelength_mid < wavelength) & (wavelength < end_left)]

        # Find start point
        if s == 0: # for MgII: use different function, that ignores the first critical point
            wavelength_start = find_criticalpoint_extended(interval_r_flux, interval_r_wavelength, interval_r_gradient)
        else:
            wavelength_start = find_criticalpoint(interval_r_flux, interval_r_wavelength, interval_r_gradient)

        if len(interval_r_wavelength) == 0: # If there are no points to right: use first point of interval
            wavelength_start = interval_wavelength[0]

        # Find end point
        if s == 0: # for MgII: use different function, that ignores the first critical point
            wavelength_end = find_criticalpoint_extended(interval_l_flux, interval_l_wavelength, interval_l_gradient)
        elif len(interval_l_wavelength) == 0: # If there are no points to left: use last point of interval
            wavelength_end = interval_wavelength[-1]
        else:
            wavelength_end = find_criticalpoint(interval_l_flux, interval_l_wavelength, interval_l_gradient)

        # If no critical points are found in the interval:

        if np.isnan(wavelength_start):
            if not np.isnan(wavelength_end): # Critical point found for end point: mirror that distance
                wavelength_start = closest(np.flip(interval_r_wavelength), wavelength_mid - (wavelength_end - wavelength_mid))
            else: # None found: take point closest to end of interval
                wavelength_start = closest(np.flip(interval_r_wavelength), end_right)

        if np.isnan(wavelength_end):
            if not np.isnan(wavelength_start): # Critical point found for start point: mirror that distance
                wavelength_end = closest(interval_l_wavelength, wavelength_mid + (wavelength_mid - wavelength_start))
            else: # None found: take point closest to end of interval
                wavelength_end = closest(interval_l_wavelength, end_left)

        # Get corresponding indices of the start and end points
        index_start = list(wavelength).index(wavelength_start)
        index_end = list(wavelength).index(wavelength_end)

        # -------- Step 3: Make continuum --------

        # Connect the start and end point by a straight line. --> y = a x + b
        if wavelength_end == wavelength_start:
            slope = 0.0
        else:
            slope = (smoothflux[index_end] - smoothflux[index_start]) / (wavelength_end - wavelength_start)
        
        intercept = smoothflux[index_start] - slope * wavelength_start

        #test_wavelength = np.linspace(wavelength_start, wavelength_end, 100)
        #test_continuum = continuum(test_wavelength)


        # -------- Step 4: Compute Pseudo-Equivalent Widths (EW) --------

        # Define the interval to look at: all points between start and end point of spectral line
        EWinterval_flux = smoothflux[(wavelength_start < wavelength) & (wavelength < wavelength_end)]
        EWinterval_wavelength = np.array(wavelength[(wavelength_start < wavelength) & (wavelength < wavelength_end)])
        EWinterval_continuum = continuum(EWinterval_wavelength, slope, intercept)


        if len(EWinterval_wavelength) == 0 or len(EWinterval_wavelength) == 1 or np.any(EWinterval_continuum == 0.0):
            # No points? EW = 0
            EW = 0.0
        else:
            # Make an array of delta_wavelength. This is the width of the bars.
            Delta_wavelength = np.append(np.diff(EWinterval_wavelength), np.diff(EWinterval_wavelength)[-1])
            # Obtain the area by multiplying the height ( = flux - continuum) by the width
            EW = np.sum((EWinterval_flux - EWinterval_continuum) / EWinterval_continuum * Delta_wavelength)

        # Add the found EW to the vector
        final_vector.append(EW)

    return final_vector


# -------------------------------

# Load the data and extract the important columns
# spectra = pd.read_pickle('../data/sdss/FinalTable_Nikki.pkl')

# -------------------------------

# Compute the spectral line vectors for all the data

def get_spectrallines(df_fluxes, df_source_info, df_wavelengths, from_sp, to_sp, save):
    """
    get_spectrallines()

    Takes a fluxes from DataFrame with spectra, and computes the area under curve at spectral lines
    to get a magnitude for each spectral line of interest (hard coded list of areas of interest)

    Parameters 
    ----------
    df_fluxes : pandas.DataFrame
        A table containing only fluxes and the corresponding objid as the first column

    df_source_info : pandas.DataFrame
        table containing all additional data about a source

    df_wavelengths : pandas.DataFrame
        table containing only a list of wavelengths that all sources share.

    from_sp : int
        The index of spectrum from which the spectral lines are calculated. Only used for the filename
        at saving

    to_sp : int
        The index of spectrum to which the spectral lines are calculated. Only used for the filename
            at saving

    save: When True, saves the resulting DataFrame
            When False, doesn't save the DataFrame

    Returns
    -------
    df : pd.DataFrame
        A pandas DataFrame with 2 columns.

        columns:	'spectral_lines',
                    'objid'
    """

    fluxes = np.delete(df_fluxes.values, 0, axis=1) # remove objids
    print(f'fluxes = {fluxes}')

    wavelengths = df_wavelengths.values.flatten()
    objid = df_source_info['objid'].to_numpy()
    z = df_source_info['z'].to_numpy()

    # Create lists for the 2 columns: the spectral lines vector and the objID list
    speclines_vector = []
    speclines_objid = []

    # Loop over all the sources in the data file: get for each one the vector with spectral lines
    m = 0
    for n in tqdm(range(len(df_source_info)), desc='Computing Spectral Lines: '):
        try:
            vector = spectrallines_1source(np.array(fluxes[n]), np.array(wavelengths), z[n])
            speclines_vector.append(vector)
            speclines_objid.append(objid[n])

        except:
            m += 1
            # print("Something went wrong with the spectral lines! At iteration ", n)
            speclines_vector.append(np.nan)
            speclines_objid.append(objid[n])

    # Merge the two columns together in a data frame
    df = {}
    df['objid'] = speclines_objid
    df['spectral_lines'] = speclines_vector
    #df['class'] = specclass
    df = pd.DataFrame(df)

    if save:
        filename = 'data/sdss/spectral_lines/spectral_lines_' + str(from_sp) + '_' + str(to_sp) + '.pkl'
        df.to_pickle(filename)

    print("There were ", m, " errors.")

    return df


# This part is for the visualisation
# Sort the spectra on the different classes: QSO, stars or galaxies

def visualize_results():
	qso_z = z[specclass == "QSO"]
	qso_fluxlist = fluxes[specclass == "QSO"]
	qso_wavelength = wavelength[specclass == "QSO"]

	star_z = z[specclass == "STAR"]
	star_fluxlist = fluxes[specclass == "STAR"]
	star_wavelength = wavelength[specclass == "STAR"]
	star_z_err = z_err[specclass == "STAR"]

	galaxy_z = z[specclass == "GALAXY"]
	galaxy_fluxlist = fluxes[specclass == "GALAXY"]
	galaxy_wavelength = wavelength[specclass == "GALAXY"]

	n = 2
	#print("Lambda_emit = ", 5400 / (1 + galaxy_z[n]))
	#print("Lambda_emit = ", 5550 / (1 + galaxy_z[n]))
	#print("Lambda_emit = ", 4350 / (1 + galaxy_z[n]))
	#print("Lambda_emit = ", 4125 / (1 + galaxy_z[n]))

	smooth_flux = apply_gaussian_filter(galaxy_fluxlist[n], sigma=4)
	delta = 80

	# Make smoother spectra
	qso_smoothflux = apply_gaussian_filter(qso_fluxlist[n], sigma=4)
	star_smoothflux = apply_gaussian_filter(star_fluxlist[n], sigma=4)
	gal_smoothflux = apply_gaussian_filter(galaxy_fluxlist[n], sigma=4)


	plt.figure(0, figsize=(14,6))
	plt.title("Quasars")
	plt.plot(qso_wavelength[n], qso_fluxlist[n], lw=0.3, color="gray")
	plt.plot(qso_wavelength[n], qso_smoothflux, '.', color="black", ms=1)
	plt.axvline(x=5895 * (1 + qso_z[n]), ls='--', color="red")
	print(5895 * (1 + qso_z[n]))
	#plt.axvline(x=midpoint, color="red")
	#plt.plot(qso_wavelength[n][start], qso_smoothflux[start], "*", ms=10, color="blue")
	#plt.plot(qso_wavelength[n][end], qso_smoothflux[end], "*", ms=10, color="blue")
	colors = ["pink", "C4", "C5", "C6", "C0", "C1", "C2", "C3", "C4", "C0"]
	for j in range(7):
		plt.axvline(x=wl_qso[j] * (1 + qso_z[n]), lw=1, color=colors[j])
		plt.axvline(x=wl_qso[j] * (1 + qso_z[n]) - delta, ls='--', lw=1, color=colors[j])
		plt.axvline(x=wl_qso[j] * (1 + qso_z[n]) + delta, ls='--', lw=1, color=colors[j])
		#plt.axvline(x=wl_qso[j], ls='--', lw=1, color=colors[j])
	plt.grid(which='minor', alpha=0.2)
	plt.grid(which='major', alpha=0.5)
	plt.xticks(np.arange(2000,9000,500))
	#plt.xlim(4500, 8000)
	#plt.ylim(0,40)
	#plt.savefig("Quasar_spectrallines2.png", dpi=500)


	plt.figure(2, figsize=(14,6))
	plt.title("Galaxies")
	plt.plot(galaxy_wavelength[n], galaxy_fluxlist[n], lw=0.3, color="gray")
	plt.plot(galaxy_wavelength[n], smooth_flux, '.', color="black", ms=1)
	plt.axvline(x=midpoint, color="red")
	plt.plot(galaxy_wavelength[n][start], gal_smoothflux[start], "*", ms=10, color="blue")
	plt.plot(galaxy_wavelength[n][end], gal_smoothflux[end], "*", ms=10, color="blue")
	plt.plot(test_wav, test_cont, '--', color="brown")
	#plt.plot(galaxy_wavelength[n], 5 * np.gradient(smooth_flux, galaxy_wavelength[n]) + 30, '--', lw=1)
	#plt.plot(galaxy_wavelength[n], 10 * np.gradient(np.gradient(smooth_flux, galaxy_wavelength[n]), galaxy_wavelength[n]) + 30, '--', lw=1)
	colors = ["C0", "C1", "C2", "C3", "C4", "C3", "C6", "C2", "C9"]
	for j in range(9):
		plt.axvline(x=wl_gal[j] * (1 + galaxy_z[n]), lw=1, color=colors[j])
		plt.axvline(x=wl_gal[j] * (1 + galaxy_z[n]) - delta, ls='--', lw=1, color=colors[j])
		plt.axvline(x=wl_gal[j] * (1 + galaxy_z[n]) + delta, ls='--', lw=1, color=colors[j])
		#plt.axvline(x=wl_qso[j], ls='--', lw=1, color=colors[j])

	plt.grid()
	plt.xticks(np.arange(4000,9000,250))
	#plt.ylim(0,1)
	#plt.xlim(4000, 6750)

	plt.figure(1, figsize=(14,6))
	plt.title("Stars")
	plt.plot(star_wavelength[n], star_fluxlist[n], lw=0.3, color="gray")
	plt.plot(star_wavelength[n], star_smoothflux, color="black", ms=1)
	colors = ["C0", "C1", "C2", "C3", "C4"]
	for j in range(5):
		plt.axvline(x=wl_star[j] * (1 + star_z[n]), lw=1, color=colors[j])
		plt.axvline(x=wl_star[j] * (1 + star_z[n]) - delta, ls='--', lw=1, color=colors[j])
		plt.axvline(x=wl_star[j] * (1 + star_z[n]) + delta, ls='--', lw=1, color=colors[j])
		#plt.axvline(x=wl_qso[j], ls='--', lw=1, color=colors[j])
	plt.grid()
	plt.xticks(np.arange(2000,9000,250))
	plt.xlim(3750, 8000)

	plt.show()

def main():
    # The spectral lines of interest
    speclines = [2799, 3727, 3934, 3968, 4101, 4304, 4342, 4861, 4960, 5008, 5175, 5895, 6565, 6716]
    speclines_name = ['MgII_em', 'OII_em', 'CAIIH_ab', 'CAIIK_ab', 'Hdelta_ab', 'Gband_ab',
                      'Hgamma_em', 'Hbeta_em', 'OIII_em', 'OIII_em', 'Mg_ab', 'NaI_ab', 'Halpha_em', 'S2_em']

    df_fluxes =  pd.read_hdf('data/sdss/preprocessed/50-100_original_fluxes.h5', key='fluxes')
    df_source_info = pd.read_hdf('data/sdss/preprocessed/50-100_original_fluxes.h5', key='spectral_data')
    df_wavelengths = pd.read_hdf('data/sdss/preprocessed/50-100_original_fluxes.h5', key='wavelengths')
    df_spectral_lines = get_spectrallines(df_fluxes=df_fluxes,
                                          df_source_info=df_source_info,
                                          df_wavelengths=df_wavelengths,
                                          from_sp=50000,
                                          to_sp=100000,
                                          save=True)

if __name__ == "__main__":
	main()