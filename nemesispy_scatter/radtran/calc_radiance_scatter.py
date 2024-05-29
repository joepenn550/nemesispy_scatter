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
from nemesispy_scatter.radtran.calc_planck import calc_planck
from nemesispy_scatter.radtran.calc_tau_gas import calc_tau_gas
from nemesispy_scatter.radtran.calc_tau_cia import calc_tau_cia
from nemesispy_scatter.radtran.calc_tau_rayleigh import calc_tau_rayleigh, rayleighls
from nemesispy_scatter.radtran.scatter import scloud11wave
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

@jit(nopython=True, parallel=False, cache=os.environ.get("USE_NUMBA_CACHE") == 'True')
def calc_radiance(wave_grid, U_layer, P_layer, T_layer, VMR_layer, A_layer, PARA_layer, phase_arr,
                  k_gas_w_g_p_t, k_wave_grid, P_grid, T_grid, del_g, ScalingFactor, R_plt, solspec,
                  k_cia, ID, cia_nu_grid, cia_frac_grid, cia_T_grid, dH, emiss_ang, sol_ang, aphi, lta, xextnorms, mu, wtmu, 
                  IRAY=4, INORMAL=0, ISPACE=1, f_flag=True, fours=-1):
    """
    Calculate emission spectrum using the correlated-k method.

    Parameters
    ----------
    wave_grid : ndarray
        Wavelengths (um) grid for calculating spectra.
    U_layer : ndarray
        Surface density of gas particles in each layer.
    P_layer : ndarray
        Atmospheric pressure grid.
    T_layer : ndarray
        Atmospheric temperature grid.
    VMR_layer : ndarray
        Array of volume mixing ratios for NGAS.
    A_layer : ndarray
        Array of aerosol densities for NMODES.
    PARA_layer : ndarray
        Atmospheric para-H2 fraction grid.
    k_gas_w_g_p_t : ndarray
        k-coefficients.
    P_grid : ndarray
        Pressure grid on which the k-coefficients are pre-computed.
    T_grid : ndarray
        Temperature grid on which the k-coefficients are pre-computed.
    del_g : ndarray
        Quadrature weights of the g-ordinates.
    ScalingFactor : ndarray
        Scale factors for line-of-sight.
    R_plt : float
        Planetary radius.
    solspec : ndarray
        Stellar spectra.

    Returns
    -------
    spectrum : ndarray
        Output spectrum.
    new_fours : int
        New value for 'fours'.
    """

    # Reorder atmospheric layers from top to bottom
    ScalingFactor = ScalingFactor[::-1]
    P_layer = P_layer[::-1]
    T_layer = T_layer[::-1]
    U_layer = U_layer[::-1]
    VMR_layer = VMR_layer[::-1, :]
    A_layer = A_layer[::-1, :]
    PARA_layer = PARA_layer[::-1]
    dH = dH[::-1]

    # Record array dimensions
    NGAS, NWAVEK, NG, NPRESS, NTEMP = k_gas_w_g_p_t.shape
    NWAVE = len(wave_grid)
    NLAYER = len(P_layer)
    NVMR = VMR_layer.shape[1]
    NMODES = A_layer.shape[1]

    # Initiate arrays to record total optical paths
    tau_total_w_g_l = np.zeros((NWAVE, NG, NLAYER))

    # Collision induced absorption optical path
    tau_cia = calc_tau_cia(wave_grid, k_cia, ISPACE, ID, U_layer, T_layer, P_layer,
                           VMR_layer, PARA_layer, dH, cia_nu_grid, cia_T_grid, cia_frac_grid, INORMAL)
    
    # Rayleigh scattering optical path
    if IRAY == 0:
        tau_rayleigh = np.zeros((NWAVE, NLAYER))
    elif IRAY == 1:
        tau_rayleigh = calc_tau_rayleigh(wave_grid, U_layer, ISPACE)
    elif IRAY == 4: 
        qh2, qhe, qnh3, qch4 = [np.zeros(NLAYER) for _ in range(4)]
        for iVMR in range(NVMR):
            if ID[iVMR] == 39:  qh2[:] = VMR_layer[:, iVMR]
            if ID[iVMR] == 40:  qhe[:] = VMR_layer[:, iVMR]
            if ID[iVMR] == 11:  qnh3[:] = VMR_layer[:, iVMR]
            if ID[iVMR] == 6:   qch4[:] = VMR_layer[:, iVMR]
        
        heoverh2, ch4overh2, nh3mix = qhe/qh2, qch4/qh2, qnh3
        tau_rayleigh = rayleighls(wave_grid, heoverh2, ch4overh2, nh3mix, U_layer, ISPACE)
    
    # Dust scattering optical path 
    xexts, xscats = phase_arr[:, :, 0].copy(), phase_arr[:, :, 1].copy()
    xscats /= xextnorms[:, None]
    xexts /= xextnorms[:, None]
    
    tau_dust_m_ext = np.zeros((NMODES, NWAVE, NLAYER))
    tau_dust_m_scat = np.zeros((NMODES, NWAVE, NLAYER))
    lfrac = np.zeros((NWAVE, NMODES, NLAYER))  

    for imode in range(NMODES):
        for iwave in range(NWAVE):
            tau_dust_m_ext[imode, iwave] = xexts[imode, iwave] * A_layer[:, imode]
            tau_dust_m_scat[imode, iwave] = xscats[imode, iwave] * A_layer[:, imode]

    tau_dust_ext = np.sum(tau_dust_m_ext, axis=0)
    tau_dust_scat = np.sum(tau_dust_m_scat, axis=0)

    for imode in range(NMODES):
        for iwave in range(NWAVE):
            for ilayer in range(NLAYER):
                if tau_dust_scat[iwave,ilayer]>0:
                    lfrac[iwave,imode,ilayer] = tau_dust_m_scat[imode,iwave,ilayer]/tau_dust_scat[iwave,ilayer]

    tau_gas = calc_tau_gas(k_gas_w_g_p_t, k_wave_grid, wave_grid, 
                           P_layer, T_layer, VMR_layer, U_layer, P_grid, T_grid, del_g, lta)
    
    omegas = np.zeros((NWAVE, NG, NLAYER))

    # Merge all different opacities
    for iwave in range(NWAVE):
        for ilayer in range(NLAYER):
            for ig in range(NG):
                tau_total_w_g_l[iwave, ig, ilayer] = (tau_gas[iwave, ig, ilayer] +
                                                      tau_cia[iwave, ilayer] +
                                                      tau_rayleigh[iwave, ilayer] +
                                                      tau_dust_ext[iwave, ilayer])
                omegas[iwave, ig, ilayer] = (tau_rayleigh[iwave, ilayer] +
                                             tau_dust_scat[iwave, ilayer]) / tau_total_w_g_l[iwave, ig, ilayer]

    # Scale to the line-of-sight opacities
    tau_total_w_g_l *= ScalingFactor
    tau_rayleigh *= ScalingFactor

    # Defining the units of the output spectrum / divide by stellar spectrum
    xfac = np.pi * 4. * np.pi * (R_plt * 1e2)**2 / solspec[:]

    spec_w_g = np.zeros((NWAVE, NG))

    # Add radiation from below deepest layer
    radground = calc_planck(wave_grid, T_layer[-1], ispace=ISPACE)
    bnu = np.zeros((NWAVE,NLAYER))
    for ilayer in range(NLAYER):
        bnu[:,ilayer] = calc_planck(wave_grid,T_layer[ilayer],ispace=ISPACE)
    
    radg = (radground * xfac / xfac * np.ones((5, NWAVE))).T
    nf = int(emiss_ang / 3) if fours == -1 else fours
    new_fours = 0
    for ig in prange(NG):
        spec_w_g[:, ig], new_spec_fours = scloud11wave(phasarr=phase_arr, radg=radg, 
                                                       sol_ang=sol_ang, emiss_ang=emiss_ang, 
                                                       solar=solspec, aphi=aphi, lowbc=1, galb=0., 
                                                       mu1=mu, wt1=wtmu, nmu=len(mu), nf=nf, igdist=1, 
                                                       vwaves=wave_grid, bnu=bnu, tau=tau_total_w_g_l[:, ig], 
                                                       tauray=tau_rayleigh, omegas=omegas[:, ig], 
                                                       nlay=NLAYER, ncont=NMODES, nphi=100, iray=IRAY, 
                                                       lfrac=lfrac, raman=False, f_flag=f_flag)
        new_fours = max(new_spec_fours, new_fours)

    spectrum = np.sum(spec_w_g * del_g, axis=1)

    return spectrum, new_fours

