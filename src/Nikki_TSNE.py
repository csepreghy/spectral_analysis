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


# Load the data and extract the important columns
spectra = pd.read_pickle('../data/sdss/speclines_test.pkl')

print(len(spectra))

print(spectra['spectral_lines'][0])

print(spectra.head())


objid = spectra.get_values()[:,0]
speclines = np.array(list(spectra.get_values()[:,1]))
specclass = spectra.get_values()[:,2]

print(speclines.shape)
print(specclass.shape)

# Make into data frame
speclines_name = ['MgII', 'OII', 'CaIIH', 'CaIIK', 'H_delta', 'G_band',
                      'H_gamma', 'H_beta', 'OIII_1', 'OIII_2', 'Mg', 'NaI', 'H_alpha', 'S2']
df = pd.DataFrame(speclines,columns=speclines_name)
df['class'] = specclass
#df['label'] = df['y'].apply(lambda i: str(i))

print(df)

# Try PCA

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

print(len(df))
print(len(speclines))


pca = PCA(n_components=3)
pca_result = pca.fit_transform(speclines)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1]
df['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))



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
plt.savefig("../plots/PCA_2comp_speclines.png", dpi=200)

# 3D
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

cmap = plt.cm.get_cmap('Accent')

rgba = cmap(0.5)
print(rgba)
legend_elements = [Line2D([0], [0], marker='o', label='GALAXY', color='w', markerfacecolor='#666666', markersize=10),
                   Line2D([0], [0], marker='o', label='QSO', color='w', markerfacecolor='#7fc87f', markersize=10),
                   Line2D([0], [0], marker='o', label='STAR', color='w', markerfacecolor='#386cb0', markersize=10)]
ax.legend(handles=legend_elements)
plt.savefig("../plots/PCA_3comp_speclines.png", dpi=200)

plt.show()