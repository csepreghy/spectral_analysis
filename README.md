# Spectral Classification of Astronomical Objects Using Neural Networks

This project initiated in the Big Data Analysis course at the University of Copenhagen (KU) by Nikki Arendse, Zoe Ansari and Cecilie Hede and Andrew Chepreghy.

Since then it has evolved to a thesis work by Andrew Chepreghy with the supervision of Adriano Agnello.

This work focuses on classifying astronomical objects such as stars, galaxies and quasars based on the spectrum of their light using different neural network architectures. There are large amounts of data collected about astronomical objects by telescopes of wide field surveys, such as the Sloan Digital Sky Survey (SDSS). The way algorithmic classification is done today is by looping through spectral templates and picking one that fits a given spectrum best. In this work I explore more efficient methods to classify SDSS spectra, and re-assess the criteria used in the SDSS template classification.

## Data

### Quasar:
<img src="https://raw.githubusercontent.com/csepreghy/spectral-analysis/master/plots/spectrum_quasar_plotify.png" width="800px" />

### Star:
<img src="https://raw.githubusercontent.com/csepreghy/spectral-analysis/master/plots/star_quasar_plotify.png" width="800px" />

### Galaxy:
<img src="https://raw.githubusercontent.com/csepreghy/spectral-analysis/master/plots/spectrum_galaxy_plotify.png" width="800px" />

The chronological order in which the downloading, data preprocessing and classification processes take place: 

1) Get coordinates from query ---------------------------------------------
2) Download data ----------------------------------------------------------
3) Merge spectra with table containing meta information -------------------
4) Filter Out Spectra with not enough values ------------------------------
5) Cut off values from the sides to have the same range for all spectra ---
6) Create Continuum that has Gaussian smoothing ---------------------------
7) Get spectral lines -----------------------------------------------------
8) Merge spectral lines with the continuum to one table -------------------
9) Merge all data ---------------------------------------------------------
10) Run the ML algorithms -------------------------------------------------


The project is run under the [DARK Cosmology Centre](https://dark.nbi.ku.dk/), [Univesity of Copenhagen](https://www.ku.dk/english/) with the assistance of [Adriano Agnello](https://www.linkedin.com/in/adriano-agnello/).
