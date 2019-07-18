import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, filters, feature
import sys

# -------------------------------

from plotify import Plotify
plotify = Plotify()

def apply_gaussian_filter(fluxes, sigma):
  return filters.gaussian(image=fluxes, sigma=sigma)

# -------------------------------

# Useful website: http://www.stat.cmu.edu/tr/tr828/tr828.pdf

# The spectral lines
qso_lines = ['MgII_em', 'Hgamma_em', 'Hbeta_em', 'OIII_em', 'OIII_em', 'Halpha_em', 'OII_em']
star_lines = ['Halpha_ab', 'Hbeta_ab', 'Hgamma_ab', 'Hdelta_ab', 'NaI_ab']
gal_lines = ['Na_ab', 'Mg_ab', 'Halpha_em', 'S2_em', 'Hbeta_em', 'Gband_ab', 'CAIIH_ab', 'CAIIK_ab', 'OII_em']#, 'balmer_break']

wl_qso = [2799, 4342, 4861, 4960, 5008, 6565, 3727]
wl_star = [6565, 4861, 4340, 4101, 5896]
wl_gal = [5893, 5175, 6565, 6716, 4861, 4304, 3933.7, 3968] #3727, 4000]

# -------------------------------

# Load the data and extract the important columns
spectra = pd.read_pickle('data/sdss/FinalTable.pkl')

flux_list = spectra.get_values()[:,0]
wavelength = spectra.get_values()[:,1]
objid = spectra.get_values()[:,2]
specclass_bytes = spectra.get_values()[:,4]
z = spectra.get_values()[:,7]
z_err = spectra.get_values()[:,5]
dec = spectra.get_values()[:,6]
ra = spectra.get_values()[:,8]
#print(spectra.iloc[0])


# Convert the specclass bytes into strings
specclass = []
for i in specclass_bytes:
    specclass.append(i.decode("utf-8"))
specclass = np.array(specclass)


# Sort the spectra on the different classes: QSO, stars or galaxies
qso_z = z[specclass == "QSO"]
qso_fluxlist = flux_list[specclass == "QSO"]
qso_wavelength = wavelength[specclass == "QSO"]

star_z = z[specclass == "STAR"]
star_fluxlist = flux_list[specclass == "STAR"]
star_wavelength = wavelength[specclass == "STAR"]
star_z_err = z_err[specclass == "STAR"]

galaxy_z = z[specclass == "GALAXY"]
galaxy_fluxlist = flux_list[specclass == "GALAXY"]
galaxy_wavelength = wavelength[specclass == "GALAXY"]

# -------------------------------

# Choose spectrum to display (and shift back to emitted wavelength)
n = 4
print("Lambda_emit = ", 4200 / (1 + galaxy_z[n]))
print("Lambda_emit = ", 5550 / (1 + galaxy_z[n]))
print("Lambda_emit = ", 4350 / (1 + galaxy_z[n]))
print("Lambda_emit = ", 5400 / (1 + qso_z[n]))



# Plot quasars
n = 100
qso_smoothflux = apply_gaussian_filter(qso_fluxlist[n], sigma=4)

_, ax = plotify.plot(
  x=qso_wavelength[n],
  y=qso_smoothflux,
  title="Spectrum of a Quasar",
  xlabel="Wavelength",
  ylabel="Fluxes",
  show_plot=False,
  xmin=3500,
  xmax=9500,
  figsize=(14,8)
)

colors = plotify.get_colors()

color_list = [
  colors['cyan'],
  colors['red'],
  colors['blue'],
  colors['purple'],
  colors['green'],
  colors['pink'],
  colors['orange'],
]


for j in range(7):
  ax.axvline(x=wl_qso[j] * (1 + qso_z[n]), lw=1, color=color_list[j])

plt.xticks(np.arange(3500, 9500, 500))
# plt.savefig("plots/spectrum_quasar_plotify.png", dpi=200)
plt.savefig(('plots/spectrum_quasar_plotify'), facecolor=plotify.background_color, dpi=180)
plt.show()


# Plot stars
n = 2
star_smoothflux = apply_gaussian_filter(star_fluxlist[n], sigma=4)
plt.figure(1, figsize=(12,6))
#plt.title("Stars")
#plt.plot(star_wavelength[n], star_fluxlist[n], lw=0.3, color="gray")
plt.plot(star_wavelength[n], star_smoothflux, color="white", ms=1)
colors = ["#ffc43d", "#7dff3d", "#3dbeff", "#dc7cff", "#ff59bf", "#ff5454"]
for j in range(5):
    plt.axvline(x=wl_star[j] * (1 + star_z[n]), lw=1, color=colors[j])
    #plt.axvline(x=wl_qso[j], ls='--', lw=1, color=colors[j])
#plt.grid()
plt.xticks(np.arange(2000,9000,250))
plt.xlim(4000, 7000)
plt.ylim(30,120)
plt.xlabel(r'Wavelength ($\AA$)')
plt.ylabel(r'Flux ($10^{-17} erg cm^{-2} s^{-1} \AA^{-1}$)', labelpad=10)
# plt.savefig("../plots/Spectrum_star_plotify.png", dpi=500)

# Plot galaxies
n = 5
gal_smoothflux = apply_gaussian_filter(galaxy_fluxlist[n], sigma=3)
plt.figure(2, figsize=(14,6))
#plt.title("Galaxies")
#plt.plot(galaxy_wavelength[n], galaxy_fluxlist[n], lw=0.3, color="#bcbcbc")
plt.plot(galaxy_wavelength[n], gal_smoothflux, color="white", ms=1)
#plt.plot(galaxy_wavelength[n], 5 * np.gradient(smooth_flux, galaxy_wavelength[n]) + 30, '--', lw=1)
#plt.plot(galaxy_wavelength[n], 10 * np.gradient(np.gradient(smooth_flux, galaxy_wavelength[n]), galaxy_wavelength[n]) + 30, '--', lw=1)
colors = ["#ffc43d", "#7dff3d", "#3dbeff", "#dc7cff", "#ff59bf", "#ff5454", "#3dbeff", "#dc7cff", "C9"]
for j in range(8):
    plt.axvline(x=wl_gal[j] * (1 + galaxy_z[n]), lw=1, color=colors[j])
    #plt.axvline(x=wl_qso[j], ls='--', lw=1, color=colors[j])
#plt.grid()
plt.xticks(np.arange(4000,9000,250))
plt.xlabel(r'Wavelength ($\AA$)')
plt.ylabel(r'Flux ($10^{-17} erg cm^{-2} s^{-1} \AA^{-1}$)', labelpad=10)
plt.ylim(0,20)
plt.xlim(4250, 7000)
# plt.savefig("../plots/spectrum_galaxy_plotify.png", dpi=500)




# Spectral line
n = 2
star_smoothflux = apply_gaussian_filter(star_fluxlist[n], sigma=1.5)
plt.figure(3, figsize=(14,6))
#plt.title("Stars")
#plt.plot(star_wavelength[n], star_fluxlist[n], lw=0.3, color="gray")
plt.plot(star_wavelength[n], star_smoothflux, color="white", ms=1)
colors = ["#ffc43d", "#7dff3d", "#3dbeff", "#dc7cff", "#ff59bf", "#ff5454"]
plt.axvline(x=wl_star[3] * (1 + star_z[n]), lw=2, color="#3dbeff", zorder=1)
#plt.axvline(x=wl_qso[j], ls='--', lw=1, color=colors[j])
#plt.grid()
plt.xticks(np.arange(2000,9000,250))
plt.xlim(4000, 4200)
plt.ylim(30,140)
plt.xlabel(r'Wavelength ($\AA$)')
plt.ylabel(r'Flux ($10^{-17} erg cm^{-2} s^{-1} \AA^{-1}$)', labelpad=10)
# plt.savefig("../plots/spectralline_plotify.png", dpi=500)
