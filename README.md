# Spectral Classification of Astronomical Objects Using Deep Neural Networks

This project initiated in the Big Data Analysis course at the University of Copenhagen (KU) by Nikki Arendse, Zoe Ansari and Cecilie Hede and Andrew Chepreghy.

Since then it has evolved to a thesis work by Andrew Chepreghy with the supervision of Adriano Agnello.

This or

The aim to create automatic spectrum classification for stellar objects with machine learning that would would save time for researchers not having to classify these objects by eye.

An example spectrum of a all 3 major classes with the added spectral lines are shown in the images below.

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
