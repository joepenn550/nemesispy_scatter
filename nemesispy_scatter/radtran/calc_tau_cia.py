#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Calculate collision-induced-absorption optical path.
"""
import numpy as np
from numba import jit
from nemesispy_scatter.common.info_cia import wnn, wnh, nn_abs, nh_abs

@jit(nopython=True)
def bilinear_xy(Q, x1, x2, y1, y2, x, y):
    fxy1 = ((x2 - x + 1e-30) / (x2 - x1 + 2e-30)) * Q[0] + ((x - x1 + 1e-30) / (x2 - x1 + 2e-30)) * Q[1]
    fxy2 = ((x2 - x + 1e-30) / (x2 - x1 + 2e-30)) * Q[2] + ((x - x1 + 1e-30) / (x2 - x1 + 2e-30)) * Q[3]
    return ((y2 - y + 1e-30) / (y2 - y1 + 2e-30)) * fxy1 + ((y - y1 + 1e-30) / (y2 - y1 + 2e-30)) * fxy2

@jit(nopython=True)
def trilinear_interpolation(grid, x_values, y_values, z_values, x_array, y_array, z_array):
    """
    Performs trilinear interpolation on a 3D grid for arrays of x, y, and z coordinates.
    Points outside the grid are assigned a value of 0.

    :param grid: 3D array of grid values.
    :param x_values: 1D array of x-axis values in the grid.
    :param y_values: 1D array of y-axis values in the grid.
    :param z_values: 1D array of z-axis values in the grid.
    :param x_array: 1D array of x-coordinates for interpolation.
    :param y_array: 1D array of y-coordinates for interpolation.
    :param z_array: 1D array of z-coordinates for interpolation.
    :return: 3D array of interpolated values.
    """
    result = np.zeros((z_array.size, x_array.size))

    for k in range(z_array.size):
        for i in range(x_array.size):
            x = x_array[i]
            if grid.shape[0] == 1:
                x = 0
            y = y_array[i]
            z = z_array[k]

            if x < x_values[0] or x > x_values[-1] \
            or y < y_values[0] or y > y_values[-1] \
            or z < z_values[0] or z > z_values[-1]:
                result[k, i] = 0
            else:
                ix = np.searchsorted(x_values, x) - 1
                iy = np.searchsorted(y_values, y) - 1
                iz = np.searchsorted(z_values, z) - 1

                ix = max(min(ix, grid.shape[0] - 2), 0)
                if ix == 0 and grid.shape[0] == 1:
                    ix = -1

                iy = max(min(iy, grid.shape[1] - 2), 0)
                iz = max(min(iz, grid.shape[2] - 2), 0)

                x1, x2 = x_values[ix], x_values[ix + 1]
                y1, y2 = y_values[iy], y_values[iy + 1]
                z1, z2 = z_values[iz], z_values[iz + 1]

                Q000, Q100, Q010, Q110 = grid[ix, iy, iz], grid[ix+1, iy, iz], grid[ix, iy+1, iz], grid[ix+1, iy+1, iz]
                Q001, Q101, Q011, Q111 = grid[ix, iy, iz+1], grid[ix+1, iy, iz+1], grid[ix, iy+1, iz+1], grid[ix+1, iy+1, iz+1]

                fz1 = bilinear_xy(np.array([Q000, Q100, Q010, Q110]), x1, x2, y1, y2, x, y)
                fz2 = bilinear_xy(np.array([Q001, Q101, Q011, Q111]), x1, x2, y1, y2, x, y)

                result[k, i] = ((z2 - z + 1e-30) / (z2 - z1 + 2e-30)) * fz1 + ((z - z1 + 1e-30) / (z2 - z1 + 2e-30)) * fz2
    return result


@jit(nopython=True, cache = False)
def calc_tau_cia(wave_grid, K_CIA, ISPACE,
    ID, TOTAM, T_layer, P_layer, VMR_layer, PARA_layer, DELH,
    cia_nu_grid, TEMPS, FRACS, INORMAL=0):
    """
    Calculates
    Parameters
    ----------
    wave_grid : ndarray
        Wavenumber (cm-1) or wavelength array (um) at which to compute
        CIA opacities.
    ID : ndarray
        Gas ID
    # ISO : ndarray
    #     Isotop ID.
    VMR_layer : TYPE
        DESCRIPTION.
    ISPACE : int
        Flag indicating whether the calculation must be performed in
        wavenumbers (0) or wavelength (1)
    K_CIA(NPAIR,NTEMP,NWAVE) : ndarray
         CIA cross sections for each pair at each temperature level and wavenumber.
    cia_nu_grid : TYPE
        DESCRIPTION.
    INORMAL : int


    Returns
    -------
    tau_cia_layer(NWAVE,NLAY) : ndarray
        CIA optical depth in each atmospheric layer.
    """
      
    NPAIR = K_CIA.shape[0]
    NLAY,NVMR = VMR_layer.shape
    ISO = np.zeros((NVMR))

    # mixing ratios of the relevant gases
    qh2 = np.zeros((NLAY))
    qhe = np.zeros((NLAY))
    qn2 = np.zeros((NLAY))
    qch4 = np.zeros((NLAY))
    qco2 = np.zeros((NLAY))

    # get mixing ratios from VMR grid
    for iVMR in range(NVMR):
        if ID[iVMR] == 39: # hydrogen
            qh2[:] += VMR_layer[:,iVMR]
        if ID[iVMR] == 40: # helium
            qhe[:] += VMR_layer[:,iVMR]
        if ID[iVMR] == 22: # nitrogen
            qn2[:] += VMR_layer[:,iVMR]
        if ID[iVMR] == 6: # methane
            qch4[:] += VMR_layer[:,iVMR]
        if ID[iVMR] == 2: # co2
            qco2[:] += VMR_layer[:,iVMR]
            
    XLEN = DELH * 1.0e2 # cm
    TOTAM = TOTAM * 1.0e-4 # cm-2

    P0=101325
    T0=273.15
    AMAGAT = 2.68675E19 #mol cm-3
    KBOLTZMANN = 1.381E-23
    MODBOLTZA = 10.*KBOLTZMANN/1.013

    tau = (P_layer/P0)**2 * (T0/T_layer)**2 * DELH
    height1 = P_layer * MODBOLTZA * T_layer

    height = XLEN * 1e2
    amag1 = TOTAM /height/AMAGAT
    tau = height*amag1**2

    AMAGAT = 2.68675E19 #mol cm-3
    amag1 = TOTAM / XLEN / AMAGAT # number density
    tau = XLEN*amag1**2# optical path

    if ISPACE == 0: 
        WAVEN = wave_grid
    elif ISPACE == 1:
        WAVEN = 1.e4/wave_grid
        isort = np.argsort(WAVEN)
        WAVEN = WAVEN[isort] 

    NWAVEC = len(wave_grid)
    tau_cia_layer = np.zeros((NWAVEC,NLAY))
    
    k_cia = np.zeros((NWAVEC,NPAIR,NLAY))

    
    for ipair in range(NPAIR):
        k_cia[:,ipair,:] = trilinear_interpolation(K_CIA[ipair], FRACS, TEMPS, cia_nu_grid, PARA_layer, T_layer, WAVEN)
    for ilay in range(NLAY):  
        if len(FRACS)==1:
            #Combining the CIA absorption of the different pairs (included in .cia file)
            sum1 = np.zeros(NWAVEC)
            if INORMAL==0: # equilibrium hydrogen (1:1)
                sum1[:] = sum1[:] + k_cia[:,0,ilay] * qh2[ilay] * qh2[ilay] \
                    + k_cia[:,1,ilay] * qhe[ilay] * qh2[ilay]
            elif INORMAL==1: # normal hydrogen (3:1)
                sum1[:] = sum1[:] + k_cia[:,2,ilay] * qh2[ilay] * qh2[ilay]\
                    + k_cia[:,3,ilay] * qhe[ilay] * qh2[ilay]

            sum1[:] = sum1[:] + k_cia[:,4,ilay] * qh2[ilay] * qn2[ilay]
            sum1[:] = sum1[:] + k_cia[:,5,ilay] * qn2[ilay] * qch4[ilay]
            sum1[:] = sum1[:] + k_cia[:,6,ilay] * qn2[ilay] * qn2[ilay]
            sum1[:] = sum1[:] + k_cia[:,7,ilay] * qch4[ilay] * qch4[ilay]
            sum1[:] = sum1[:] + k_cia[:,8,ilay] * qh2[ilay] * qch4[ilay]

            # look up CO2-CO2 CIA coefficients (external)
            """
            TO BE DONE
            """
            k_co2 = sum1*0
            # k_co2 = co2cia(WAVEN)

            sum1[:] = sum1[:] + k_co2[:] * qco2[ilay] * qco2[ilay]

            #Look up N2-N2 NIR CIA coefficients
            k_n2 = np.interp(WAVEN,wnn,nn_abs)
            k_n2[WAVEN<np.min(wnn)] = 0.0
            k_n2[WAVEN>np.max(wnn)] = 0.0
            sum1[:] = sum1[:] + k_n2[:] * qn2[ilay] * qn2[ilay] * 1e-5

            #Look up N2-H2 NIR CIA coefficients
            k_n2h2 = np.interp(WAVEN,wnh,nh_abs)
            k_n2h2[WAVEN<np.min(wnh)] = 0.0
            k_n2h2[WAVEN>np.max(wnh)] = 0.0
            sum1[:] = sum1[:] + k_n2h2[:] * qn2[ilay] * qh2[ilay] * 1e-5
            # TO BE DONE


        else:
            sum1 = np.zeros(NWAVEC)
            sum1[:] = sum1[:] + k_cia[:,0,ilay] * qh2[ilay] * qh2[ilay]
            sum1[:] = sum1[:] + k_cia[:,1,ilay] * qh2[ilay] * qhe[ilay]

        tau_cia_layer[:,ilay] = sum1[:] * tau[ilay]     
    if ISPACE==1:
        tau_cia_layer[:,:] = tau_cia_layer[isort,:]
    return tau_cia_layer
