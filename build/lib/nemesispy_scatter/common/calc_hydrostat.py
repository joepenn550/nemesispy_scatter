#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Use hydrostatic equilibrium to find altitudes given pressures, temperatures
and mean molecular weights.
"""
import numpy as np
from numba import jit, njit
from nemesispy_scatter.common.constants import G, K_B

@jit(nopython=True)
def calc_grav_simple(h, M_plt, R_plt):
    """
    Calculates the gravitational acceleration at altitude h on a planet.

    Parameters
    ----------
    h : real
        Altitude.
        Unit: m
    M_plt : real
        Planet mass.
        Unit: kg
    R_plt : real
        Planet radius.
        Unit: m

    Returns
    -------
    g : real
        Gravitational acceleration.
        Unit: ms^-2
    """
    g = G*M_plt/(R_plt+h)**2
    return g

@njit
def thetagc(lat, e):
    return np.arctan(np.tan(lat) / (e**2))

@njit
def legpol(n, z):
    pol = np.array([1, z])
    if n == 0 or n == 1:
        return pol[n]
    else:
        for i in range(2, n+1):
            pol_next = ((2 * i - 1) * z * pol[1] - (i - 1) * pol[0]) / i
            pol = np.array([pol[1], pol_next])
        return pol[1]


@njit
def calc_grav(params, H):
    
    Grav=6.672E-11        
        
    mass = params[0]*Grav*1e6
    j2 = params[1]
    j3 = params[2]
    j4 = params[3]
    r0 = params[4]*1e2
    flat = params[5]
    rot = params[6]
    
    lat_in = params[-1]
    
    pi = np.pi
    xellip =  1 / (1-flat)
    xradius = r0
    h = H/1e3
    xcoeff = np.array([j2,j3,j4])
    xgm = mass
    
    xomega = 2*np.pi/rot
    
    lat = 2 * pi * lat_in / 360.
    latc = thetagc(lat, xellip)
    slatc = np.sin(latc)
    clatc = np.cos(latc)

    Rr = np.sqrt(clatc**2 + (xellip**2 * slatc**2))
    r = (xradius + h*1e5) / Rr
    radius = (xradius / Rr) * 1e-5
    pol = np.zeros(6)
    for i in range(1, 7):
        pol[i-1] = legpol(i, slatc)

    g = 1.
    for i in range(1, 4):
        g -= (2*i+1) * Rr**(2 * i) * xcoeff[i-1] * pol[2*i-1]

    g = (g * xgm / r**2) - (r * xomega**2 * clatc**2)

    gtheta = 0.
    for i in range(1, 4):
        gtheta -= (4 * i**2 * Rr**(2 * i) * xcoeff[i-1] * (pol[2*i-2] - slatc * pol[2*i-1]) / clatc)

    gtheta = (gtheta * xgm / r**2) + (r * xomega**2 * clatc * slatc)

    g = np.sqrt(g**2 + gtheta**2) * 0.01
    
#     print(g)
    return g
 
@jit(nopython=True, cache = False)
def calc_hydrostat(P, T, mmw, M_plt, R_plt, params, H=np.array([])):
    """
    Calculates an altitude profile from given pressure, temperature and
    mean molecular weight profiles assuming hydrostatic equilibrium.


    Parameters
    ----------
    P : ndarray
        Pressure profile
        Unit: Pa
    T : ndarray
        Temperature profile
        Unit: K
    mmw : ndarray
        Mean molecular weight profile
        Unit: kg
    M_plt : real
        Planetary mass
        Unit: kg
    R_plt : real
        Planetary radius
        Unit: m
    H : ndarray
        Altitude profile to be adjusted
        Unit: m

    Returns
    -------
    adjusted_H : ndarray
        Altitude profile satisfying hydrostatic equlibrium.
        Unit: m

    """
    # Note number of profile points and set up a temporary height profile
    NPRO = len(P)
    if len(H)==0:
        H = np.linspace(0,1e6,NPRO)

    # First find level closest ot zero altitude
    ialt = (np.abs(H - 0.0)).argmin()
    alt0 = H[ialt]
#     if ( (alt0>0.0) & (ialt>0)):
#         ialt = ialt -1

    # iterate until hydrostatic equilibrium
    xdepth = 2
    adjusted_H = H
    dummy_H = np.zeros(NPRO)
    while xdepth > 1:

        dummy_H = adjusted_H

        # Calculate the atmospheric model depth
        atdepth = dummy_H[-1] - dummy_H[0]
        # Calculate the gravity at each altitude level
        gravity =  calc_grav(params,dummy_H)
        # Calculate the scale height
        R = 8.31
        AVOGAD = 6.022045E23
        scale = R*T/(mmw*AVOGAD*gravity)

        if ialt > 0 and ialt < NPRO-1 :
            dummy_H[ialt] = 0.0

        # nupper = NPRO - ialt - 1
        for i in range(ialt+1, NPRO):
            sh = 0.5 * (scale[i-1] + scale[i])
            dummy_H[i] = dummy_H[i-1] - sh * np.log(P[i]/P[i-1])

        for i in range(ialt-1,-1,-1):
            sh = 0.5 * (scale[i+1] + scale[i])
            dummy_H[i] = dummy_H[i+1] - sh * np.log(P[i]/P[i+1])

        atdepth1 = dummy_H[-1] - dummy_H[0]
        xdepth = 100.*np.abs((atdepth1-atdepth)/atdepth)
        adjusted_H = dummy_H
    return adjusted_H