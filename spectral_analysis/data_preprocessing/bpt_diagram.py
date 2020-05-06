import numpy as np
import pandas as pd
from spectral_analysis.plotify import Plotify
import matplotlib.pyplot as plt

def plot_bpt_diagram(df_source_info):
    print(f'{df_source_info}')
    df_source_info = df_source_info.drop(df_source_info[df_source_info['Flux_NII_6547'] < 0].index).reset_index()
    print(df_source_info)

    Hb = df_source_info['Flux_Hb_4861']
    Ha = df_source_info['Flux_Ha_6562']
    
    O3_1 = df_source_info['Flux_OIII_4958']
    O3_2 = df_source_info['Flux_OIII_5006']
    O3 = np.add(O3_1, O3_2)
    
    N2_1 = df_source_info['Flux_NII_6547']
    N2_2 = df_source_info['Flux_NII_6583']
    N2 = np.add(N2_1, N2_2)

    yvals = np.log10(np.divide(O3, Hb))
    xvals = np.log10(np.divide(N2, Ha))

    plotify = Plotify(theme='ugly')
    _, ax = plotify.get_figax()
    ax.scatter(xvals, yvals, s=0.2, alpha=0.8, c=plotify.c_orange, label='Sources')
    ax.set_xlabel(r'$ log([NII] / H\alpha$)', fontsize=16)
    ax.set_ylabel(r'$ log([OIII] / H\beta$)', fontsize=16)

    X = np.linspace(-1.5,0.3)
    Y = (0.61/( X  - 0.47  )) + 1.19

    # Schawinski+07 --------------------------------------
    X3 = np.linspace(-0.180,1.5)
    Y3 = 1.05*X3 + 0.45

    # Kauffmann+03 ---------------------------------------
    Xk = np.linspace(-1.5,0.)
    Yk = 0.61/(Xk -0.05) + 1.3

    # Regions --------------------------------------------
    ax.plot(X,   Y, '-' , color=plotify.c_blue, lw=2, label='Kewley+01') # Kewley+01
    ax.plot(X3, Y3, '-', color=plotify.c_blue, lw=2, label='Schawinski+07') # Schawinski+07
    ax.plot(Xk, Yk, '--', color=plotify.c_blue, lw=2, label='Kauffmann+03') # Kauffmann+03

    ax.set_title('BPT Diagram', fontsize=18)
    plt.legend()
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.5, 1)
    plt.show()
    

def main():
    df_fluxes = pd.read_hdf('data/sdss/preprocessed/balanced_with_emission_lines.h5', key='fluxes')
    df_source_info = pd.read_hdf('data/sdss/preprocessed/balanced_with_emission_lines.h5', key='source_info')
    df_wavelengths = pd.read_hdf('data/sdss/preprocessed/balanced_with_emission_lines.h5', key='wavelengths')


    plot_bpt_diagram(df_source_info)

if __name__ == "__main__":
    main()