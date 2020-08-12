## Spectral Classification of Astronomical Objects Using Deep Neural Networks

This project initiated in the Big Data Analysis course at the University of Copenhagen (KU) by Nikki Arendse, Zoe Ansari and Cecilie Hede and Andrew Chepreghy.

Since then it has evolved to a thesis work by Andrew Chepreghy with the supervision of Adriano Agnello.

This work focuses on classifying astronomical objects such as stars, galaxies and quasars based on the spectrum of their light using different neural network architectures. There are large amounts of data collected about astronomical objects by telescopes of wide field surveys, such as the Sloan Digital Sky Survey (SDSS). The way algorithmic classification is done today is by looping through spectral templates and picking one that fits a given spectrum best. In this work I explore more efficient methods to classify SDSS spectra, and re-assess the criteria used in the SDSS template classification.

## Data

There are 3 main classes in the dataset 2 of which have further subclasses that are later considered in subclass classification. Below there are 3 representative examples of spectra from each main class. The complete dataset contained 96,115 spectra consisting of 53,825 galaxies (56.0%), 32,488 quasars (33.8%) and 9,802 stars (10.2%).

### Star

<p align="center">
  <img src="https://raw.githubusercontent.com/csepreghy/spectral-analysis/master/plots/thesis_plots/example_star.png" width="600px" />
</p>

### Galaxy

<p align="center">
  <img src="https://raw.githubusercontent.com/csepreghy/spectral-analysis/master/plots/thesis_plots/example_galaxy.png" width="600px" />
</p>

### Quasar

<p align="center">
  <img src="https://raw.githubusercontent.com/csepreghy/spectral_analysis/master/plots/thesis_plots/examlpe_qso.png" width="600px" />
</p>

## Data Preprocessing

The chronological order in which the downloading, data preprocessing and classification processes take place: 

1) Get coordinates from SQL query -------------------------------------------
2) Download spectral data from SDSS -----------------------------------------
3) Merge spectra with table containing meta information ---------------------
4) Filter Out Spectra with insufficient number of values --------------------
5) Cut off values from the edges to have the same range for all spectra -----
6) (optional) Create continuum that has Gaussian smoothing ------------------
7) Get spectral lines -------------------------------------------------------
8) Merge spectral lines and raw spectral data -------------------------------
9) Merge all data into HDF5 fiels -------------------------------------------
10) Train and test a given neural network -----------------------------------

## Neural Networks
I propose two neural network architectures for this project. A Convolutional Neural Network (CNN) coupled with hyperparameter optimization implemented with Keras Tuner and a Mixed Input Neural Network (MINN).

## CNN with Keras-Tuner hypertuning
This approach focuses only on the fluxes of a spectrum (raw datapoints), using a 1 dimensional CNN that extracts successively larger visual features in a hierarchical set of layers. The intuition behind which is that since an astronomer is able to recognize visual patterns by only looking at the spectrum, a CNN in principle should be able to do the same. To maximize performance Keras Tuner was used for hypertuning. The network architecture is shwon below.

<p align="center">
  <img src="https://raw.githubusercontent.com/csepreghy/spectral_analysis/master/plots/thesis_plots/Andy-squared-equals-neural-network-modified.png" width="400px"/>
</p>

### Results
64,000 spectra were randomly subsampled on which 40 trials were run with different hyperparameters, the best of which resulted in a training accuracy: 0.999 and a **test accuracy: 0.989**.

## Mixed-input Neural Network
The other pruposed neural network architecture is a mixed-input neural network (MINN). It incorporates a 1D CNN for the fluxes (visual features) described above and a feed forward neural network for numerical features such as redshift, Petrosian magnitudes and spectral lines. These features are provided by SDSS except spectral lines which are computed in this project. The architecture is shown below:

<p align="center">
  <img src="https://raw.githubusercontent.com/csepreghy/spectral_analysis/master/plots/thesis_plots/Neural-Andy-Layer.png" width="400px"/>
</p>

The project is run under the [DARK Cosmology Centre](https://dark.nbi.ku.dk/), [Univesity of Copenhagen](https://www.ku.dk/english/) with the assistance of [Adriano Agnello](https://www.linkedin.com/in/adriano-agnello/).
