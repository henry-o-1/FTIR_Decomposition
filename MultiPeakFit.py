from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from main import Amide_I_Region, Second_Derivative_Spectrum

#read from txt file and convert to data frame of floats
df = pd.read_csv('FTIR_data',sep="\t", header=None)
df = df.astype('str')
df.columns = ['k', 'Abs']
df = df.astype('float')

#filters to Amide I region and sets data frame to match
df = df.query('k < 1700 & k > 1600')

k = df.k.to_numpy()
Abs = df.Abs.to_numpy()

Amide_I = Amide_I_Region()
def plot_gauss(theta):
    #theta = (peak loc, width, height)
    k = np.array([np.linspace(1600,1700,300)])

    gaussian_array = np.zeros((1,300))

    for i in theta:
        sigma = i[1] / np.sqrt(2)
        gauss = i[2] * np.exp(-((k - i[0]) ** 2) / (2 * (sigma ** 2)))
        gaussian_array = np.append(gaussian_array,gauss,axis=0)
    gaussian_array = np.delete(gaussian_array,0,axis=0)
    return gaussian_array

def construction(theta):
    construction = np.sum(plot_gauss(theta),axis=0)
    return construction

def baseline_subtract(x,amide_data):
    baseline = (Amide_I.baseline_param(k,Abs)[0] * x) + Amide_I.baseline_param(k,Abs)[1]
    return amide_data - baseline

def residuals(theta0):
    theta = np.reshape(theta0,(-1,3))
    #first we have to get from size of data array to theoretical/guess array
    guess_data = np.interp(df.k,np.linspace(1600,1700,300),construction(theta))
    return baseline_subtract(df.k,df.Abs) - guess_data

"""
#Peak Parameters via usr input
def Usr_Input():
    Input_Peak_Loc = []
    Input_Peak_Width = []
    Input_Peak_Height = []
    while True:
        Peak_Loc = input("Input peak location (Type 'Y' if finished entering peak parameters): ")
        if Peak_Loc == "Y":
            break
        Peak_Width = input("Input peak width: ")
        Peak_Height = input("Input peak height: ")

        Input_Peak_Loc.append(Peak_Loc)
        Input_Peak_Width.append(Peak_Width)
        Input_Peak_Height.append(Peak_Height)

    Input_Peak_Loc = np.array([float(i) for i in Input_Peak_Loc])
    Input_Peak_Width = np.array([float(i) for i in Input_Peak_Width])
    Input_Peak_Height = np.array([float(k) for k in Input_Peak_Height])
    return np.vstack((Input_Peak_Loc,Input_Peak_Width,Input_Peak_Height))
theta0_ravel = Usr_Input().T.ravel()
"""

#Peak Parameter Definition NO USR Input
peaksA = np.array([1610,1617,1635,1646,1653,1661,1669,1676,1685,1690])
widthsA = np.ones_like(peaksA) * 5
heights = np.ones_like(peaksA) * 0.01
theta0_ravel = np.vstack((peaksA, widthsA, heights)).T.ravel()

#raveled array must be fed to residuals function, it is reshaped in the function
#To test, everything else can be fed theta

theta0 = np.reshape(theta0_ravel,(-1,3))

#Defines construction
sum_peaks = construction(theta0)

#Plot initial guess data
fig, axs = plt.subplots(3,1,sharex=True,figsize=(8,8))
axs[0].plot(np.linspace(1600,1700,300),plot_gauss(theta0).T)
axs[1].plot(df.k,baseline_subtract(k,Abs))
axs[1].plot(np.linspace(1600,1700,300),sum_peaks)
axs[2].plot(df.k,residuals(theta0_ravel))

axs[0].set_title('Original Peak Guesses')
axs[1].set_title('Data and Construction')
axs[1].legend(['Amide I Data','Construction'],loc='upper right')
axs[2].set_title('Residuals')

#Fits Parameters, using Levenberg-Marquardt method
fit_params = least_squares(residuals,theta0_ravel,method='lm').x

#Feeds residuals in raveled form to return fitted residuals, and reshapes parameters to be fed to construction
fit_residuals = residuals(fit_params)
fit_params = np.reshape(fit_params,(-1,3))
#Defines construction and individual peaks. Transpose fit_peaks because it returns n guesses rows by k columns
fit_sum = construction(fit_params)
fit_peaks = plot_gauss(fit_params).T

fig, axs = plt.subplots(3,1,sharex=True,figsize=(8,8))
axs[0].plot(np.linspace(1600,1700,300),fit_peaks)
axs[1].plot(df.k,baseline_subtract(df.k,df.Abs))
axs[1].plot(np.linspace(1600,1700,300),fit_sum)
axs[2].plot(df.k,fit_residuals)

axs[0].set_title('Fitted Peaks')
axs[1].set_title('Data and Fitted Construction')
axs[1].legend(['Amide I Data','Construction'])
axs[2].set_title('Fitted Residuals')
axs[2].set_xlabel('Wavenumber (cm^-1)')

def peak_label(fitted_params):
    #function takes fitted parameters and makes them convenient for annotation
    fitted_params_transpose = fitted_params.T
    # arrays of transpose are 1: peak loc 2: peak width 3: peak height
    Peak_Locs = fitted_params_transpose[0]
    Peak_Heights = fitted_params_transpose[2]

    #Location and Height can be put into array, signify position on fitted chart
    return np.array(list(zip(Peak_Locs,Peak_Heights)))

#Prepares annotation for plot which shows peak wavenumber
for x, y in peak_label(fit_params):
    label = "{:.0f}".format(x)
    axs[0].annotate(label, (x, y), xytext=(0, 5), textcoords='offset points', ha="center",
                    arrowprops=dict(arrowstyle="-", connectionstyle="arc3"))

plt.show()

def peak_integration(domain,fitted_params):
    integral = np.trapz(plot_gauss(fitted_params),domain,axis=1)
    total_area = np.sum(integral)
    component_percent = (integral / total_area) * 100
    peak_wavenumber = fitted_params.T[0]

    #Rounding for prettier format
    integral = ["%.6f" % i for i in integral]
    total_area = "{:.6f}".format(total_area)
    component_percent = ["%.2f" % i for i in component_percent]
    peak_wavenumber = ["%.2f" % i for i in peak_wavenumber]

    return [integral, component_percent, total_area,peak_wavenumber]

integral_data = peak_integration(np.linspace(1600,1700,300),fit_params)

for i, integration in enumerate(integral_data[0]):
    print("Peak %s: %s cm^-1: %s (%s%%)" % (i, integral_data[3][i], integration, integral_data[1][i]))

#Total Integration
print("Total Area: %s" % integral_data[2])