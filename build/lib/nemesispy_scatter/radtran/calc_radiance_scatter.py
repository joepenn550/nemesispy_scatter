#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Routines to calculate thermal emission spectra from a planetary atmosphere
using the correlated-k method to combine gaseous opacities.
We inlcude collision-induced absorption from H2-H2, H2-he, H2-N2, N2-Ch4, N2-N2,
Ch4-Ch4, H2-Ch4 pairs and Rayleigh scattering from H2 molecules and
He molecules.

As of now the routines are fully accelerated using numba.jit.
"""
import os
import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
from nemesispy_scatter.radtran.calc_planck import calc_planck
from nemesispy_scatter.radtran.calc_tau_gas import calc_tau_gas#,calc_tau_gas_lta
from nemesispy_scatter.radtran.calc_tau_cia import calc_tau_cia
from nemesispy_scatter.radtran.calc_tau_rayleigh import calc_tau_rayleigh, rayleighls
from nemesispy_scatter.radtran.scatter import makephase, scloud11wave
from nemesispy_scatter.common.constants import K_B, N_A


# import inspect
# import sys
# def recompile_nb_code():
#     this_module = sys.modules[__name__]
#     module_members = inspect.getmembers(this_module)

#     for member_name, member in module_members:
#         if hasattr(member, 'recompile') and hasattr(member, 'inspect_llvm'):
#             member.recompile()
# recompile_nb_code()

@jit(nopython=True, parallel = False, cache = os.environ.get("USE_NUMBA_CACHE")=='True')
def calc_radiance(wave_grid, U_layer, P_layer, T_layer, VMR_layer,A_layer,PARA_layer,phase_arr,
    k_gas_w_g_p_t, k_wave_grid, P_grid, T_grid, del_g, ScalingFactor, R_plt, solspec,
    k_cia, ID, cia_nu_grid, cia_frac_grid, cia_T_grid, dH, emiss_ang, sol_ang, aphi, lta, xextnorms, mu, wtmu, 
                  IRAY=4, INORMAL=0, ISPACE=1, f_flag = True, fours = -1):
    """
    Calculate emission spectrum using the correlated-k method.

    Parameters
    ----------
    wave_grid(NWAVE) : ndarray
        Wavelengths (um) grid for calculating spectra.
    U_layer(NLAYER) : ndarray
        Surface density of gas particles in each layer.
        Unit: no. of particle/m^2
    P_layer(NLAYER) : ndarray
        Atmospheric pressure grid.
        Unit: Pa
    T_layer(NLAYER) : ndarray
        Atmospheric temperature grid.
        Unit: K
    VMR_layer(NLAYER,NGAS) : ndarray
        Array of volume mixing ratios for NGAS.
        Has dimension: NLAYER x NGAS
    A_layer(NLAYER,NGAS) : ndarray
        Array of aerosol densities for NMODES.
        Has dimension: NLAYER x NMODES
    PARA_layer(NLAYER) : ndarray
        Atmospheric para-h2 fraction grid.
    k_gas_w_g_p_t(NGAS,NWAVE,NG,NPRESSK,NTEMPK) : ndarray
        k-coefficients.
        Has dimension: NGAS x NWAVE x NG x NPRESSK x NTEMPK.
    P_grid(NPRESSK) : ndarray
        Pressure grid on which the k-coeff's are pre-computed.
        We want SI unit (Pa) here.
    T_grid(NTEMPK) : ndarray
        Temperature grid on which the k-coeffs are pre-computed. In Kelvin
    del_g : ndarray
        Quadrature weights of the g-ordinates.
    ScalingFactor(NLAYER) : ndarray
        Scale stuff to line of sight
    R_plt : real
        Planetary radius
        Unit: m
    solspec : ndarray
        Stellar spectra, used when the unit of the output is in fraction
        of stellar irradiance.

        Stellar flux at planet's distance (W cm-2 um-1 or W cm-2 (cm-1)-1)

    Returns
    -------
    spectrum : ndarray
        Output spectrum (W cm-2 um-1 sr-1)
    """#
    
    # Reorder atmospheric layers from top to bottom
    ScalingFactor = ScalingFactor[::-1]
    P_layer = P_layer[::-1] # layer pressures (Pa)
    T_layer = T_layer[::-1] # layer temperatures (K)
    U_layer = U_layer[::-1] # layer absorber amounts (no./m^2)
    VMR_layer = VMR_layer[::-1,:] # layer volume mixing ratios
    A_layer = A_layer[::-1,:] # aerosol 
    PARA_layer = PARA_layer[::-1] # para-h2 fraction
    dH = dH[::-1] # lengths of each layer
    # Record array dimensions
    NGAS, NWAVEK, NG, NPRESS, NTEMP = k_gas_w_g_p_t.shape
    NWAVE = len(wave_grid)
    NLAYER = len(P_layer)
    NVMR = VMR_layer.shape[1]
    NMODES = A_layer.shape[1]

    # Initiate arrays to record total optical paths
    tau_total_w_g_l = np.zeros((NWAVE,NG,NLAYER))

    # Collision induced absorptioin optical path (NWAVE x NLAYER)
    tau_cia = calc_tau_cia(wave_grid=wave_grid,K_CIA=k_cia,ISPACE=ISPACE,
        ID=ID,TOTAM=U_layer,T_layer=T_layer,P_layer=P_layer,VMR_layer=VMR_layer,PARA_layer=PARA_layer,
        DELH=dH,cia_nu_grid=cia_nu_grid,TEMPS=cia_T_grid,FRACS=cia_frac_grid,INORMAL=INORMAL)
    
    
    
    # Rayleigh scattering optical path (NWAVE x NLAYER)
    if IRAY == 0:
        tau_rayleigh = np.zeros((NWAVE,NLAYER))
    
    elif IRAY == 1:
        tau_rayleigh = calc_tau_rayleigh(wave_grid=wave_grid,U_layer=U_layer,ISPACE=ISPACE)
        
    elif IRAY == 4: 
        qh2 = np.zeros(NLAYER)
        qhe = np.zeros(NLAYER)
        qnh3 = np.zeros(NLAYER)
        qch4 = np.zeros(NLAYER)
        for iVMR in range(NVMR):
            if ID[iVMR] == 39: # hydrogen
                qh2[:] = VMR_layer[:,iVMR]
            if ID[iVMR] == 40: # helium
                qhe[:] = VMR_layer[:,iVMR]
            if ID[iVMR] == 11: # ammonia
                qnh3[:] = VMR_layer[:,iVMR]
            if ID[iVMR] == 6: # methane
                qch4[:] = VMR_layer[:,iVMR]
        
        heoverh2 = qhe/qh2
        ch4overh2 = qch4/qh2
        nh3mix = qnh3
        tau_rayleigh = rayleighls(wave_grid,heoverh2,ch4overh2,nh3mix,U_layer,ISPACE=ISPACE)
    
    # Dust scattering optical path 
    xexts = phase_arr[:,:,0].copy()
    xscats = phase_arr[:,:,1].copy()#*phase_arr[:,:,0].copy() 
    for imode in range(NMODES):
        xscats[imode] = xscats[imode]/xextnorms[imode]
        xexts[imode] = xexts[imode]/xextnorms[imode]
    

    tau_dust_m_ext = np.zeros((NMODES,NWAVE,NLAYER))
    tau_dust_m_scat = np.zeros((NMODES,NWAVE,NLAYER))
    lfrac = np.zeros((NWAVE,NMODES, NLAYER))  

    for imode in range(NMODES):
        for iwave in range(NWAVE):
            for ilayer in range(NLAYER):
                tau_dust_m_ext[imode,iwave,ilayer] = xexts[imode,iwave]*A_layer[ilayer,imode]
                tau_dust_m_scat[imode,iwave,ilayer] = xscats[imode,iwave]*A_layer[ilayer,imode]

    tau_dust_ext = np.sum(tau_dust_m_ext,axis=0)
    tau_dust_scat = np.sum(tau_dust_m_scat,axis=0)           




    for imode in range(NMODES):
        for iwave in range(NWAVE):
            for ilayer in range(NLAYER):
                if tau_dust_scat[iwave,ilayer]>0:

                    lfrac[iwave,imode,ilayer] = tau_dust_m_scat[imode,iwave,ilayer]/tau_dust_scat[iwave,ilayer]

    tau_gas = calc_tau_gas(k_gas_w_g_p_t, k_wave_grid, wave_grid, P_layer, T_layer, VMR_layer, U_layer,
        P_grid, T_grid, del_g, lta)
    
    omegas = np.zeros((NWAVE,NG,NLAYER))

    # Merge all different opacities
    for iwave in range(NWAVE):
        for ilayer in range(NLAYER):
            for ig in range(NG):
                tau_total_w_g_l[iwave,ig,ilayer] = tau_gas[iwave,ig,ilayer] \
                    + tau_cia[iwave,ilayer] \
                    + tau_rayleigh[iwave,ilayer] \
                    + tau_dust_ext[iwave,ilayer] \
                
                
                omegas[iwave,ig,ilayer] = (tau_rayleigh[iwave,ilayer]\
                                           + tau_dust_scat[iwave,ilayer])\
                                            /tau_total_w_g_l[iwave,ig,ilayer]
                
                
    # Scale to the line-of-sight opacities
    tau_total_w_g_l = tau_total_w_g_l*ScalingFactor
    tau_rayleigh = tau_rayleigh*ScalingFactor
    # Defining the units of the output spectrum / divide by stellar spectrum
    xfac = np.pi*4.*np.pi*(R_plt*1e2)**2./solspec[:]

    spec_w_g = np.zeros((NWAVE,NG))

    # Add radiation from below deepest layer
    radground = calc_planck(wave_grid,T_layer[-1],ispace=ISPACE)
    bnu = np.zeros((NWAVE,NLAYER))
    for ilayer in range(NLAYER):
        bnu[:,ilayer] = calc_planck(wave_grid,T_layer[ilayer],ispace=ISPACE)
    
    radg = (radground*xfac/xfac*np.ones((5,NWAVE))).transpose()
    comp = 14
#     print('CIA:',tau_cia[0,5],'GAS: ',tau_gas[0,0,5],'RAY: ', tau_rayleigh[0,5],'EXT: ', tau_dust_ext[0,5],
#           'SCAT: ', tau_dust_scat[0,5],'OMEGA: ',omegas[0,0,5],'TOTAL: ',tau_total_w_g_l[0,0,5],flush=True)


    if fours == -1:
        nf = int(emiss_ang/3)
    else:
        nf = fours
    new_fours = 0
    for ig in prange(NG):
        spec_w_g[:,ig], new_spec_fours = scloud11wave(phasarr=phase_arr,radg = radg, 
                                sol_ang = sol_ang, emiss_ang = emiss_ang, solar = solspec, aphi = aphi, lowbc = 1, galb = 0., 
                                mu1 = mu, wt1 = wtmu, nmu = len(mu), nf = nf, igdist = 1, vwaves = wave_grid, 
                                bnu = bnu, tau = tau_total_w_g_l[:,ig], 
                                tauray = tau_rayleigh, omegas = omegas[:,ig], 
                                nlay = NLAYER, ncont = NMODES, nphi = 100, iray = IRAY, lfrac = lfrac, 
                                raman = False, f_flag = f_flag)
        new_fours = max(new_spec_fours,new_fours)
    spectrum = np.zeros((NWAVE))
    for iwave in range(NWAVE):
        for ig in range(NG):
            spectrum[iwave] += spec_w_g[iwave,ig] * del_g[ig]
    return spectrum, new_fours

