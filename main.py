import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter

# read from txt file and convert to data frame of floats
df = pd.read_csv('FTIR_data', sep="\t", header=None)
df = df.astype('str')
df.columns = ['k', 'Abs']
df = df.astype('float')

# stores series k and Abs from df
k = df['k']
Abs = df['Abs']

# derivatives
dAbsdk = np.gradient(Abs, k)
d2Absdk2 = np.gradient(dAbsdk, k)
df['d2Absdk2'] = d2Absdk2  # stores second derivative in dataframe, for rel extrema

# filters to Amide I region and sets data frame to match
df = df.query('k < 1700 & k > 1600')

class Amide_I_Region:
    # Function which returns baseline Parameters
    # NOTE: Arguments must be NUMPY ARRAYS and NOT PANDAS SERIES
    def baseline_param(self, k, Abs):
        try:
            m = (Abs[-1] - Abs[0]) / (k[-1] - k[0])
            y0 = Abs[-1] + (m * (0 - k[-1]))
            return [m, y0]
        except:
            print("Make sure arguments to baseline_param are NUMPY Arrays (not pandas series)\ne.g. \"df.k\"")

class Second_Derivative_Spectrum:
    def __init__(self, extrema_order, SG_order, SG_window):
        self.extrema_order = extrema_order
        self.SG_order = SG_order
        self.SG_window = SG_window

    #Savitzky Golay filter for smoothing. Minimum window and order to preserve features
    #Typically, Window Length is 5 and order is 2. For preservation of spectrum features

    def SG_filt(self,d2Absdk2):
        Filtered_Second_Deriv = savgol_filter(d2Absdk2,window_length=self.SG_window,polyorder=self.SG_order)
        return Filtered_Second_Deriv

    #location of relative minima, order is usually 1 for specificity
    def extrema(self,filt_d2Absdk2):
        indices = argrelextrema(filt_d2Absdk2, np.less, order=self.extrema_order)
        return indices

    #Returns list of SG-smoothed relative extrema
    def minima(self,k,filt_d2Absdk2):
        k_minima = k[self.extrema(filt_d2Absdk2)]
        d2Absdk2_minima = filt_d2Absdk2[self.extrema(filt_d2Absdk2)]
        return [k_minima,d2Absdk2_minima]

    def Spectrum_Diagnostics(self,k,filt_d2Absdk2):
        print("Spectrum Diagnostics:")
        print("Extrema order: %s\nSavitzky-Golay order: %s\nSavitzky-Golay window-size: %s" % (self.extrema_order,self.SG_order,self.SG_window))
        return "Savitzky_Golay Minima Found at %s" % self.minima(k,filt_d2Absdk2)[0][::-1]


if __name__ == '__main__':
    Second_Deriv = Second_Derivative_Spectrum(1, SG_order=2, SG_window=5)
    Amide_I = Amide_I_Region()

    #Turn data frame series to numpy arrays
    k = df.k.to_numpy()
    Abs = df.Abs.to_numpy()
    d2Absdk2 = df.d2Absdk2.to_numpy()

    # plots original Amide I region (Abs vs k)
    fig, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
    axs[0].plot(k, Abs, markersize=4)
    axs[0].set_title('Amide I Region (Abs/k)')


    # plotting baseline
    k_baseline = np.linspace(1600, 1700, 101)
    y_baseline = (Amide_I.baseline_param(k,Abs)[0] * k_baseline) + Amide_I.baseline_param(k,Abs)[1]

    # plot base line, define domain and plot
    axs[0].plot(k_baseline, y_baseline)
    axs[0].set_ylabel('Absorbance [Arbitrary Units]')
    axs[0].legend(['Abs vs k', 'baseline'])

    print("A (intercept): %s\nB (slope): %s" % (Amide_I.baseline_param(k,Abs)[1], Amide_I.baseline_param(k,Abs)[0]))

    # smooth second derivative and plot it
    SG_second_deriv = Second_Deriv.SG_filt(d2Absdk2)
    axs[1].plot(k, SG_second_deriv, marker='.')
    axs[1].set_xlabel('k (cm^-1)')
    axs[1].set_ylabel('Savitzky-Golay-Filtered d2Absdk2')

    # annotate local minima for second deriv analysis
    k_minima = Second_Deriv.minima(k,SG_second_deriv)[0]
    SG_d2Absdk2_minima = Second_Deriv.minima(k,SG_second_deriv)[1]

    for x, y in zip(k_minima, SG_d2Absdk2_minima):
        label = "{:.0f}".format(x)
        axs[1].annotate(label, (x, y), xytext=(0, -10), textcoords='offset points', ha="center")

    print(Second_Deriv.Spectrum_Diagnostics(k,SG_second_deriv))

    plt.show()