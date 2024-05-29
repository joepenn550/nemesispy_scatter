#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""Routines to read pre-tabulated correlated-k look-up tables (k-tables) and
collision induced absorption opacitied files.

All k-tables are assumed to share the same wavelength grid, pressure grid,
temperature grid, g ordinates and quadrature weights.

Note that the opacities in Nemesis k-tables are multiplied by a factor of 1e20,
which is balanced in the radiative transfer routine by multiplying
absorber amounts by 1e-20.
"""
import numpy as np
from scipy.io import FortranFile
from nemesispy_scatter.common.constants import ATM

def read_tables(filepaths):
    """
    Read a list of k-tables from serveral Nemesis .kta files.

    Parameters
    ----------
    filepaths : list
        A list of strings containing filepaths to the .kta files to be read.

    Returns
    -------
    gas_id_list(NGAS) : ndarray
        Gas identifier list.
    iso_id_list(NGAS) : ndarray
        Isotopologue identifier list.
    wave_grid(NWAVEKTA) : ndarray
        Wavenumbers/wavelengths grid of the k-table.
    g_ord(NG) : ndarray
        Quadrature points on the g-ordinates
    del_g(NG) : ndarray
        Gauss quadrature weights for the g-ordinates.
        These are the widths of the bins in g-space.
    P_grid(NPRESSKTA) : ndarray
        Pressure grid on which the k-coefficients are pre-computed.
        Unit: Pa
    T_grid(NTEMPKTA) : ndarray
        Temperature grid on which the k-coefficients are pre-computed.
        Unit: Kelvin
    k_gas_w_g_p_t(NGAS,NWAVEKTA,NG,NPRESSKTA,NTEMPKTA) : ndarray
        Array storing the k-coefficients.

    Notes
    -----
    Assume the k-tables in all the kta files are computed on the same
    wavenumber/wavelength, pressure and temperature grid and with
    same g-ordinates and quadrature weights.
    """
    k_gas_w_g_p_t = []
    gas_id_list = []
    iso_id_list = []
    lta_flags = []
    for filepath in filepaths:
        if filepath[-3:] == 'kta':
            gas_id, iso_id, wave_grid, g_ord, del_g, P_grid, T_grid,\
              k_g = read_kta(filepath)
            lta_flags.append(False)
            
        else:
            gas_id, iso_id, wave_grid, g_ord, del_g, P_grid, T_grid,\
              k_g = read_lta(filepath)
            lta_flags.append(True)

        gas_id_list.append(gas_id)
        iso_id_list.append(iso_id)
        k_gas_w_g_p_t.append(k_g)
    # reformat data type to appease numba
    gas_id_list = np.float32(gas_id_list)
    iso_id_list = np.float32(iso_id_list)
    k_gas_w_g_p_t = np.float32(k_gas_w_g_p_t)
    lta_flags = np.array(lta_flags)
    return gas_id_list, iso_id_list, wave_grid, g_ord, del_g, P_grid, T_grid,\
        k_gas_w_g_p_t, lta_flags

def read_kta(filepath):
    """
    Reads a correlated-k look-up table from a .kta file in Nemesis format.

    Parameters
    ----------
    filepath : str
        The filepath to the .kta file to be read.

    Returns
    -------
    gas_id : int
        Gas identifier.
    iso_id : int
        Isotopologue identifier.
    wave_grid(NWAVEKTA) : ndarray
        Wavenumber/wavelength grid of the k-table.
    g_ord(NG) : ndarray
        Quadrature points on the g-ordinates
    del_g(NG) : ndarray
        Gaussian quadrature weights for the g-ordinates.
        These are the widths of the bins in g-space.
    P_grid(NPRESSKTA) : ndarray
        Pressure grid on which the k-coefficients are pre-computed.
        Unit: Pa
    T_grid(NTEMPKTA) : ndarray
        Temperature grid on which the k-coefficients are pre-computed.
        Unit: Kelvin
    k_w_g_p_t(NWAVEKTA,NG,NPRESSKTA,NTEMPKTA) : ndarray
        Array storing the k-coefficients.
    """
    # Open file
    if filepath[-3:] == 'kta':
        f = open(filepath,'r')
    else:
        f = open(filepath+'.kta','r')

    # Define bytes consumed by elements of table
    nbytes_int32 = 4
    nbytes_float32 = 4
    ioff = 0 # current position in the file

    # Read headers, irec0 is where ktable data starts
    irec0 = int(np.fromfile(f,dtype='int32',count=1))
    NWAVEKTA = int(np.fromfile(f,dtype='int32',count=1))
    vmin = float(np.fromfile(f,dtype='float32',count=1))
    delv = float(np.fromfile(f,dtype='float32',count=1))
    fwhm = float(np.fromfile(f,dtype='float32',count=1))
    NPRESSKTA = int(np.fromfile(f,dtype='int32',count=1))
    NTEMPKTA = int(np.fromfile(f,dtype='int32',count=1))
    NG = int(np.fromfile(f,dtype='int32',count=1))
    gas_id = int(np.fromfile(f,dtype='int32',count=1))
    iso_id = int(np.fromfile(f,dtype='int32',count=1))

    ioff += 10*nbytes_int32

    # Read g-ordinates and quadrature weights
    g_ord = np.fromfile(f,dtype='float32',count=NG)
    del_g = np.fromfile(f,dtype='float32',count=NG)
    dummy1 = np.fromfile(f,dtype='float32',count=1)
    dummy2 = np.fromfile(f,dtype='float32',count=1)

    ioff += 2*NG*nbytes_float32
    ioff += 2*nbytes_float32

    # Read temperature/pressure grid
    # Note that kta pressure grid is in ATM, conver to Pa
    P_grid = np.fromfile(f,dtype='float32',count=NPRESSKTA) * np.float32(ATM)
    T_grid = np.fromfile(f,dtype='float32',count=NTEMPKTA)
    ioff += NPRESSKTA*nbytes_float32 + NTEMPKTA * nbytes_float32

    # Read wavenumber/wavelength grid
    if delv>0.0:  # uniform grid
        vmax = delv*(NWAVEKTA-1) + vmin
        wave_grid = np.linspace(vmin,vmax,NWAVEKTA)
    else:   # non-uniform grid
        wave_grid = np.zeros((NWAVEKTA))
        wave_grid = np.fromfile(f,dtype='float32',count=NWAVEKTA)
        ioff += NWAVEKTA*nbytes_float32

    # Jump to the minimum wavenumber
    if ioff > (irec0-1)*nbytes_float32:
        raise(Exception('Error in {} : too many headers'.format(filepath)))
    ioff = (irec0-1)*nbytes_float32 # Python index starts at 0
    f.seek(ioff,0)

    # Write the k-coefficients into array form
    k_w_g_p_t = np.zeros((NWAVEKTA,NG,NPRESSKTA,NTEMPKTA))
    k_list = np.fromfile(f,dtype='float32',count=NTEMPKTA*NPRESSKTA*NG*NWAVEKTA)
    ig = 0
    for iwave in range(NWAVEKTA):
        for ipress in range(NPRESSKTA):
            for itemp in range(NTEMPKTA):
                k_w_g_p_t[iwave,:,ipress,itemp] = k_list[ig:ig+NG]
                ig = ig + NG
    k_w_g_p_t = np.float32(k_w_g_p_t)

    # close file
    f.close()
    
    T_grid = T_grid[None,:] # sneaky hack to make ltas work
    
    return gas_id, iso_id, np.float64(wave_grid), g_ord, del_g, P_grid,\
        T_grid, k_w_g_p_t

def read_cia(filepath,dnu,npara):
    """
    Parameters
    ----------
    filepath : str
        Filepath to the .tab file containing CIA information.
    dnu : real, optional
        Wavenumber interval. The default is 10.
    npara : int, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    NU_GRID(NWAVE) : ndarray
        Wavenumber array (NOTE: ALWAYS IN WAVENUMBER, NOT WAVELENGTH).
    TEMPS(NTEMP) : ndarray
        Temperature levels at which the CIA data is defined (K).
    K_CIA(NPAIR,NTEMP,NWAVE) : ndarray
         CIA cross sections for each pair at each temperature level and wavenumber.
    """
    f = FortranFile(filepath, 'r')
    
    
    if npara != 0:
        NPAIR = 2
        TEMPS = f.read_reals(dtype='float32')
        FRACS = np.abs(f.read_reals(dtype='float32'))
        K_H2H2 = f.read_reals(dtype='float32')
        K_H2HE = f.read_reals(dtype='float32')
        KCIA_list = np.vstack([K_H2H2,K_H2HE]).reshape((-1,),order='F')

    if npara == 0:
        NPAIR = 9 # 9 pairs of collision induced absorption opacities
        FRACS = np.array([0])
        TEMPS = f.read_reals( dtype='float64' )
        KCIA_list = f.read_reals( dtype='float32' )
        
    NT = len(TEMPS)
    NWAVE = int(len(KCIA_list)/NT/NPAIR/max(npara,1))
    NU_GRID = np.linspace(0,dnu*(NWAVE-1),NWAVE)
    K_CIA = np.zeros((NPAIR,max(npara,1),NT,NWAVE)) # NPAIR x NT x NWAVE

    index = 0
    for iwn in range(NWAVE):
        for itemp in range(NT):
            for ipara in range(max(npara,1)):
                for ipair in range(NPAIR):
                
                    K_CIA[ipair,ipara,itemp,iwn] = KCIA_list[index]
                    index += 1

    return NU_GRID, FRACS, TEMPS, K_CIA


def read_lta(filepath):
    if filepath[-3:] == 'lta':
        f = open(filepath,'r')
    else:
        f = open(filepath+'.lta','r')

    # Read headers, irec0 is where ktable data starts
    irec0 = int(np.fromfile(f,dtype='int32',count=1))
    npoint = int(np.fromfile(f,dtype='int32',count=1))
    vmin = float(np.fromfile(f,dtype='float32',count=1))
    delv = float(np.fromfile(f,dtype='float32',count=1))
    n_p = int(np.fromfile(f,dtype='int32',count=1))
    n_t = int(np.fromfile(f,dtype='int32',count=1))
    gas_id = int(np.fromfile(f,dtype='int32',count=1))
    iso_id = int(np.fromfile(f,dtype='int32',count=1))

    irec = 9
    
    P_grid = np.fromfile(f,dtype='float32',count=n_p) * np.float32(ATM)
    if n_t > 0:
        T_grid = np.fromfile(f,dtype='float32',count=n_t)
    else:
        T_grid = np.zeros((n_p,2))
        for i in range(n_p):
            T_grid[i] = np.fromfile(f,dtype='float32',count=-n_t)

    if delv>0.0:  # uniform grid
        vmax = delv*(npoint-1) + vmin
        wave_grid = np.linspace(vmin,vmax,npoint)
    else:   # non-uniform grid
        wave_grid = np.zeros((npoint))
        wave_grid = np.fromfile(f,dtype='float32',count=npoint)
        
    np.fromfile(f,dtype='float32',count=3) # what's this bit?

    k_w_p_t = np.fromfile(f,dtype='float32',count=npoint*n_p*np.abs(n_t)).reshape((npoint,n_p,np.abs(n_t)))

    f.close()
    
    k_w_g_p_t = k_w_p_t[:,None,:,:]
    g_ord = 0
    del_g = np.array([1.0])
    
    return gas_id, iso_id, np.float64(wave_grid), g_ord, del_g, P_grid, T_grid, k_w_g_p_t