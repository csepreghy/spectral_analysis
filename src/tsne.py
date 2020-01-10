import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sb
from matplotlib.lines import Line2D
import time as time
from sklearn.preprocessing import StandardScaler

# -----------------------------------------
# Load the data and extract the important columns

spectra = pd.read_pickle('../COMPLETE_df.pkl')
full_spectra = pd.read_pickle('../data/sdss/FinalTable_Nikki.pkl')

# -----------------------------------------

def get_PCA(spectra, dimensions, speclines="yes", continuum="yes"):
    # Get important columns from data frame
    
    z = spectra['z'].get_values()
    z_err = spectra['zErr'].get_values()
    flux_list = list(spectra['flux_list'].get_values())
    wavelength = spectra['wavelength'].get_values()[0]
    spectral_lines = list(spectra['spectral_lines'].get_values())
    specclass = spectra['class']

    print(np.shape(flux_list))


    # Input variables
    if speclines == "yes" and continuum == "yes":
      X = np.hstack((z.reshape(-1,1), z_err.reshape(-1,1), spectral_lines, flux_list))
    elif speclines == "yes" and continuum == "no":
      X = np.hstack((z.reshape(-1, 1), z_err.reshape(-1, 1), spectral_lines))
    else:
      X = np.hstack((z.reshape(-1, 1), z_err.reshape(-1, 1), flux_list))


    # Make into data frame
    speclines_name = ['MgII', 'OII', 'CaIIH', 'CaIIK', 'H_delta', 'G_band',
                          'H_gamma', 'H_beta', 'OIII_1', 'OIII_2', 'Mg', 'NaI', 'H_alpha', 'S2']
    df1 = pd.DataFrame(spectral_lines, columns=speclines_name)
    df2 = pd.DataFrame(flux_list, columns=wavelength)

    if speclines == "yes" and continuum == "yes":
      df = pd.concat([df1, df2], axis=1)
    elif speclines == "yes" and continuum == "no":
      df = df1
    else:
      df = df2


    # Make new specclass array, with numbers. QSO = 1, STAR = 2, GALAXY = 3
    classnumber = []
    for j in range(len(df)):
      if specclass[j] == "GALAXY":
        classnumber.append(3.0)
      elif specclass[j] == "STAR":
        classnumber.append(2.0)
      elif specclass[j] == "QSO":
        classnumber.append(1.0)

    df['z'] = z
    df['zErr'] = z_err
    df['class'] = specclass
    df['class_numbers'] = classnumber


    print(df.iloc[0])

    # Do PCA
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    pca = PCA(n_components=dimensions)
    pca_result = pca.fit_transform(X_std)
    df['pca-one'] = pca_result[:, 0]
    df['pca-two'] = pca_result[:, 1]
    if dimensions > 2:
        df['pca-three'] = pca_result[:, 2]
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    # -------- Make plots --------

    if dimensions == 2:
        plt.figure(figsize=(16, 10))
        sb.scatterplot(
            x="pca-one", y="pca-two",
            hue="class",
            palette=sb.color_palette("hls", 3),
            data=df,
            legend="full",
            alpha=0.3
        )
        if speclines == "yes" and continuum == "yes":
            plt.xlim(-500, 400)
            plt.ylim(-500, 3000)
            plt.savefig("../plots/PCA_scaled_" + str(dimensions) + "comp_continuum+speclines.png", dpi=300)
        elif speclines == "yes" and continuum == "no":
            plt.xlim(-100, 0)
            plt.ylim(-55, 75)
            plt.savefig("../plots/PCA_scaled_" + str(dimensions) + "comp_speclines.png", dpi=300)
        else:
            plt.xlim(-250, 400)
            plt.ylim(-400, 300)
            plt.savefig("../plots/PCA_scaled_" + str(dimensions) + "comp_continuum.png", dpi=300)

    plt.show()


    return 5.0

# -----------------------------------------  -----------------------------------------


def get_tSNE(spectra, dimensions, speclines="yes", continuum="yes"):

    # Get important columns from data frame
    z = spectra['z'].get_values()
    z_err = spectra['zErr'].get_values()
    flux_list = list(spectra['flux_list'].get_values())
    wavelength = spectra['wavelength'].get_values()[0]
    spectral_lines = list(spectra['spectral_lines'].get_values())
    specclass = spectra['class']


    # Input variables
    if speclines == "yes" and continuum == "yes":
        X = np.hstack((z.reshape(-1,1), z_err.reshape(-1,1), spectral_lines, flux_list))
    elif speclines == "yes" and continuum == "no":
        X = np.hstack((z.reshape(-1, 1), z_err.reshape(-1, 1), spectral_lines))
    else:
        X = np.hstack((z.reshape(-1, 1), z_err.reshape(-1, 1), flux_list))


    # Make into data frame
    speclines_name = ['MgII', 'OII', 'CaIIH', 'CaIIK', 'H_delta', 'G_band',
                          'H_gamma', 'H_beta', 'OIII_1', 'OIII_2', 'Mg', 'NaI', 'H_alpha', 'S2']
    df1 = pd.DataFrame(spectral_lines, columns=speclines_name)
    df2 = pd.DataFrame(flux_list, columns=wavelength)

    if speclines == "yes" and continuum == "yes":
        df = pd.concat([df1, df2], axis=1)
    elif speclines == "yes" and continuum == "no":
        df = df1
    else:
        df = df2


    # Make new specclass array, with numbers. QSO = 1, STAR = 2, GALAXY = 3
    classnumber = []
    for j in range(len(df)):
        if specclass[j] == "GALAXY":
            classnumber.append(3.0)
        elif specclass[j] == "STAR":
            classnumber.append(2.0)
        elif specclass[j] == "QSO":
            classnumber.append(1.0)

    df['z'] = z
    df['zErr'] = z_err
    df['class'] = specclass
    df['class_numbers'] = classnumber

    print(df.iloc[0])

    # Do t-SNE

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    time_start = time.time()
    tsne = TSNE(n_components=dimensions, verbose=0, perplexity=40, n_iter=1000)
    tsne_results = tsne.fit_transform(X_std)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    # -------- Make plots --------
    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]
    plt.figure(figsize=(16, 10))
    sb.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="class",
        palette=sb.color_palette("hls", 3),
        data=df,
        legend="full",
        alpha=0.3)
    if speclines == "yes" and continuum == "yes":
        plt.savefig("../plots/tSNE_scaled_" + str(dimensions) + "comp_" + str(len(spectra)) + "_continuum+speclines.png", dpi=300)
    elif speclines == "yes" and continuum == "no":
        plt.savefig("../plots/tSNE_scaled_" + str(dimensions) + "comp_" + str(len(spectra)) + "_speclines.png", dpi=300)
    else:
        plt.savefig("../plots/tSNE_scaled_" + str(dimensions) + "comp_" + str(len(spectra)) + "_continuum.png", dpi=300)

    plt.show()

    return 5.0




# -----------------------------------------  -----------------------------------------

get_tSNE(spectra, dimensions=2)




sys.exit()

z = full_spectra.get_values()[:, 7]
objid = spectra.get_values()[:,0]
speclines = list(spectra.get_values()[:,1])
specclass = spectra.get_values()[:,2]

Y = np.append(speclines, z.reshape(-1,1), axis=1)


# Make into data frame
speclines_name = ['MgII', 'OII', 'CaIIH', 'CaIIK', 'H_delta', 'G_band',
                      'H_gamma', 'H_beta', 'OIII_1', 'OIII_2', 'Mg', 'NaI', 'H_alpha', 'S2', 'z']
df = pd.DataFrame(Y, columns=speclines_name)
df['class'] = specclass


# Remove any rows that have NaNs as spectral lines vector
remove_index = []
for i in range(len(df)):
    if np.any(np.isnan(speclines[i])):
        #print("NaN! ", i)
        #print(speclines[i])
        remove_index.append(i)

df = df.drop(remove_index, axis=0)
speclines = np.delete(speclines, remove_index, axis=0)
specclass = np.delete(specclass, remove_index, axis=0)
Y = np.delete(Y, remove_index, axis=0)


# Make new specclass array, with numbers. QSO = 1, STAR = 2, GALAXY = 3
classnumber = []
for j in range(len(df)):
    if specclass[j] == "GALAXY":
        classnumber.append(3.0)
    elif specclass[j] == "STAR":
        classnumber.append(2.0)
    elif specclass[j] == "QSO":
        classnumber.append(1.0)

df['class_numbers'] = classnumber


print(df.iloc[0])
df.to_pickle('../data/sdss/speclines_0-10000.pkl')

sys.exit()


# ------- ------- PCA ------- -------

pca = PCA(n_components=3)
pca_result = pca.fit_transform(Y)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1]
df['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# ------- ------- ------- -------

# Make PCA plots
plt.figure(figsize=(16,10))
sb.scatterplot(
    x="pca-one", y="pca-two",
    hue="class",
    palette=sb.color_palette("hls", 3),
    data=df,
    legend="full",
    alpha=0.3
)
plt.xlim(-150,0)
plt.ylim(-100,100)
#plt.savefig("../plots/PCA_2comp_speclines.png", dpi=200)

"""
# Try out with 2 random spectral lines
plt.figure(figsize=(16,10))
sb.scatterplot(
    x="CaIIH", y="CaIIK",
    hue="class",
    palette=sb.color_palette("hls", 3),
    data=df,
    legend="full",
    alpha=0.3
)
plt.xlim(-150,150)
plt.ylim(-150,150)
"""

# 3D plot
ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=df["pca-one"],
    ys=df["pca-two"],
    zs=df["pca-three"],
    c=df["class_numbers"],
    cmap='Accent' )
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
ax.set_xlim(-200, 100)
ax.set_ylim(-200, 200)
ax.set_zlim(-200, 200)

legend_elements = [Line2D([0], [0], marker='o', label='GALAXY', color='w', markerfacecolor='#666666', markersize=10),
                   Line2D([0], [0], marker='o', label='QSO', color='w', markerfacecolor='#7fc87f', markersize=10),
                   Line2D([0], [0], marker='o', label='STAR', color='w', markerfacecolor='#386cb0', markersize=10)]
ax.legend(handles=legend_elements)
#plt.savefig("../plots/PCA_3comp_speclines.png", dpi=200)



# ------- ------- TSNE ------- ------- #

# Project into 2 dimensions
time_start = time.time()
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=1000)
tsne_results = tsne.fit_transform(Y)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

# Plot TSNE
df['tsne-2d-one'] = tsne_results[:,0]
df['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sb.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="class",
    palette=sb.color_palette("hls", 3),
    data=df,
    legend="full",
    alpha=0.3 )
#plt.savefig("../plots/TSNE3_ncomponents=2_verbose=0_perplexity=40_niter=1000.png", dpi=300)

# ------- ------- ------- ------- -------
# TSNE: now project into 3 dimensions

time_start = time.time()
tsne = TSNE(n_components=3, verbose=0, perplexity=40, n_iter=1000)
tsne_results = tsne.fit_transform(speclines)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
df['tsne-3d-one'] = tsne_results[:,0]
df['tsne-3d-two'] = tsne_results[:,1]
df['tsne-3d-three'] = tsne_results[:,2]


# Plot in 3D
ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
  xs=df["tsne-3d-one"],
  ys=df["tsne-3d-two"],
  zs=df["tsne-3d-three"],
  c=df["class_numbers"],
  cmap='Accent'
)
ax.set_xlabel('tsne-one')
ax.set_ylabel('tsne-two')
ax.set_zlabel('tsne-three')
#ax.set_xlim(-200, 100)
#ax.set_ylim(-200, 200)
#ax.set_zlim(-200, 200)

legend_elements = [Line2D([0], [0], marker='o', label='GALAXY', color='w', markerfacecolor='#666666', markersize=10),
                   Line2D([0], [0], marker='o', label='QSO', color='w', markerfacecolor='#7fc87f', markersize=10),
                   Line2D([0], [0], marker='o', label='STAR', color='w', markerfacecolor='#386cb0', markersize=10)]
ax.legend(handles=legend_elements)
#plt.savefig("../plots/PCA_3comp_speclines.png", dpi=200)



plt.show()