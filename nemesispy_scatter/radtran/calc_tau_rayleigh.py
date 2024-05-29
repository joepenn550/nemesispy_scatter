#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Calculate Rayleigh scattering optical path.
"""
from numba import jit
import numpy as np

@jit(nopython=True, cache = False)
def calc_tau_rayleigh(wave_grid,U_layer,ISPACE=1):
    """
    Calculate the Rayleigh scattering optical depth for Gas Giant atmospheres
    using data from Allen (1976) Astrophysical Quantities.

    Assume H2 ratio of 0.864.

    Parameters
    ----------
    wave_grid(NWAVE) : ndarray
        Wavenumber (cm-1) or wavelength array (um)
    U_layer(NLAYER) : ndarray
        Total absober amount
    ISPACE : int
        Flag indicating the spectral units
        (0) Wavenumber in cm-1 or (1) Wavelegnth (um)

    Outputs
    -------
    tau_rayleigh(NWAVE,NLAYER) : ndarray
        Rayleigh scattering optical path at each wavlength in each layer
    """
    AH2 = 13.58E-5
    BH2 = 7.52E-3
    AHe = 3.48E-5
    BHe = 2.30E-3
    fH2 = 0.864
    k = 1.37971e-23 # JK-1
    P0 = 1.01325e5 # Pa
    T0 = 273.15 # K

    NLAYER = len(U_layer)
    NWAVE = len(wave_grid)

    if ISPACE == 0:
        LAMBDA = 1./wave_grid * 1.0e-2 # converted wavelength unit to m
        x = 1.0/(LAMBDA*1.0e6)
    else:
        LAMBDA = wave_grid * 1.0e-6 # wavelength in m
        x = 1.0/(LAMBDA*1.0e6)

    # calculate refractive index
    nH2 = AH2 * (1.0+BH2*x*x)
    nHe = AHe * (1.0+BHe*x*x)

    #calculate Jupiter air's refractive index at STP (Actually n-1)
    nAir = fH2 * nH2 + (1-fH2)*nHe

    #H2,He Seem pretty isotropic to me?...Hence delta = 0.
    #Penndorf (1957) quotes delta=0.0221 for H2 and 0.025 for He.
    #(From Amundsen's thesis. Amundsen assumes delta=0.02 for H2-He atmospheres
    delta = 0.0
    temp = 32*(np.pi**3.)*nAir**2.
    N0 = P0/(k*T0)

    x = N0*LAMBDA*LAMBDA
    faniso = (6.0+3.0*delta)/(6.0 - 7.0*delta)

    # Calculate the scattering cross sections in m2
    k_rayleighj = temp*faniso/(3.*(x**2)) #(NWAVE)

    # Calculate the Rayleigh opacities in each layer
    tau_rayleigh = np.zeros((len(wave_grid),NLAYER))

    for ilay in range(NLAYER):
        tau_rayleigh[:,ilay] = k_rayleighj[:] * U_layer[ilay] #(NWAVE,NLAYER)

    return tau_rayleigh

@jit(nopython=True, cache = False)
def rayleighls(wave_grid,heoverh2,ch4overh2,nh3mix,U_layer,ISPACE=1):
    
    a = np.array([13.58e-5, 3.48e-5, 37.0e-5, 37.0e-5])
    b = np.array([7.52e-3,  2.3e-3, 12.0e-3, 12.0e-3])
    d = np.array([0.0221,   0.025,    0.0922, 0.0922])
    NLAYER = len(U_layer)
    comp = np.zeros((4,heoverh2.shape[0]))
    h2overtot = (1-nh3mix)/(1+heoverh2+ch4overh2)
    comp[0] = h2overtot
    comp[1] = heoverh2*h2overtot
    comp[2] = ch4overh2*h2overtot
    comp[3] = nh3mix
    
    loschpm3=2.687e19*1e-12
    rls = np.zeros((len(wave_grid),len(heoverh2)))
    for iwave in range(len(wave_grid)):
        if ISPACE == 1:
            wl=wave_grid[iwave]
        else:
            wl = 1e4/wave_grid[iwave]
        xc1 = np.zeros(comp.shape[1])
        sumwt = np.zeros(comp.shape[1])
        for j in range(4):
            nr = 1.0+a[j]*(1.0+b[j]/wl**2)
            xc1 += (nr**2-1)**2*comp[j]*(6+3*d[j])/(6-7*d[j])
            sumwt += comp[j]

        fact = (8.0*(np.pi**3))/(3.0*(wl**4)*(loschpm3**2))

        rls[iwave,:] = fact*1e-12*xc1/sumwt
        
    for ilay in range(NLAYER):
        rls[:,ilay] = rls[:,ilay] * U_layer[ilay] #(NWAVE,NLAYER)
    return rls
