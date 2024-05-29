#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Routines to split an atmosphere into layers and calculate the average layer
properties along a slant path.
"""
import os
import numpy as np
from numba import jit
from nemesispy_scatter.common.constants import K_B, N_A

def split(H_model, P_model, NLAYER, layer_type=1, H_0=0.0,
          planet_radius=None, custom_path_angle=0.0,
          custom_H_base=np.array([0,0]), custom_P_base=np.array([0,0])):
    """
    Splits atmospheric models into layers. Returns layer base altitudes
    and layer base pressures.
    """
    if layer_type == 1:  # split by equal log pressure intervals
        bottom_pressure = np.interp(H_0, H_model, P_model)
        P_base = np.exp(np.linspace(np.log(bottom_pressure), np.log(P_model[-1]), NLAYER+1))[:-1]
        P_model = P_model[::-1]
        H_model = H_model[::-1]
        H_base = np.interp(P_base, P_model, H_model)

    elif layer_type == 0:  # split by equal pressure intervals
        bottom_pressure = np.interp(H_0, H_model, P_model)
        P_base = np.linspace(bottom_pressure, P_model[-1], NLAYER+1)[:-1]
        P_model = P_model[::-1]
        H_model = H_model[::-1]
        H_base = np.interp(P_base, P_model, H_model)

    elif layer_type == 2:  # split by equal height intervals
        H_base = np.linspace(H_model[0] + H_0, H_model[-1], NLAYER+1)[:-1]
        P_base = np.interp(H_base, H_model, P_model)

    elif layer_type == 3:  # split by equal line-of-sight path intervals
        assert 0 <= custom_path_angle <= 90, 'Zenith angle should be in range [0,90] degrees'
        sin = np.sin(custom_path_angle * np.pi / 180)
        cos = np.cos(custom_path_angle * np.pi / 180)
        r0 = planet_radius + H_0
        rmax = planet_radius + H_model[-1]
        S_max = np.sqrt(rmax**2 - (r0 * sin)**2) - r0 * cos
        S_base = np.linspace(0, S_max, NLAYER+1)[:-1]
        H_base = np.sqrt(S_base**2 + r0**2 + 2 * S_base * r0 * cos) - planet_radius
        logP_base = np.interp(H_base, H_model, np.log(P_model))
        P_base = np.exp(logP_base)

    elif layer_type == 4:  # split by specifying input base pressures
        NLAYER = len(custom_P_base)
        P_model = P_model[::-1]
        H_model = H_model[::-1]
        H_base = np.interp(custom_P_base[::-1] * 101325, P_model, H_model)[::-1]
        P_base = custom_P_base

    elif layer_type == 5:  # split by specifying input base heights
        assert custom_H_base[0] >= H_model[0] and custom_H_base[-1] < H_model[-1], \
            'Input layer base heights out of range of atmosphere profile'
        NLAYER = len(custom_H_base)
        P_base = np.interp(custom_H_base, H_model, P_model)

    else:
        raise ValueError('Layering scheme not defined')

    return H_base, P_base

@jit(nopython=True, cache=os.environ.get("USE_NUMBA_CACHE") == 'True')
def simps(y, x):
    """
    Numerical integration using the composite Simpson's rule.
    """
    dx = x[1] - x[0]
    even = np.sum(y[2:-1:2])
    odd = np.sum(y[1::2])
    integral = (dx / 3) * (y[0] + 4 * odd + 2 * even + y[-1])
    return integral

@jit(nopython=True, cache=os.environ.get("USE_NUMBA_CACHE") == 'True')
def average(planet_radius, H_model, P_model, T_model, VMR_model, A_model, PARA_model, ID, H_base,
            path_angle, H_0=0.0):
    """
    Calculates absorber-amount-weighted average layer properties of an
    atmosphere.
    """
    NSIMPS = 101
    NLAYER = len(H_base)
    dH = np.concatenate((H_base[1:] - H_base[:-1], np.array([H_model[-1] - H_base[-1]])))
    sin = np.sin(path_angle * np.pi / 180)
    cos = np.cos(path_angle * np.pi / 180)
    r0 = planet_radius + H_0
    rmax = planet_radius + H_model[-1]
    S_max = np.sqrt(rmax**2 - (r0 * sin)**2) - r0 * cos
    S_base = np.sqrt((planet_radius + H_base)**2 - (r0 * sin)**2) - r0 * cos
    dS = np.concatenate((S_base[1:] - S_base[:-1], np.array([S_max - S_base[-1]])))
    scale = dS / dH
    Ngas = len(VMR_model[0])
    Nmodes = len(A_model[0])
    H_layer = np.zeros(NLAYER)
    P_layer = np.zeros(NLAYER)
    T_layer = np.zeros(NLAYER)
    U_layer = np.zeros(NLAYER)
    PARA_layer = np.zeros(NLAYER)
    VMR_layer = np.zeros((NLAYER, Ngas))
    A_layer = np.zeros((NLAYER, Nmodes))

    for ilayer in range(NLAYER):
        S0 = S_base[ilayer]
        S1 = S_base[ilayer + 1] if ilayer < NLAYER - 1 else S_max
        S_int = np.linspace(S0, S1, NSIMPS)
        H_int = np.sqrt(S_int**2 + r0**2 + 2 * S_int * r0 * cos) - planet_radius
        P_int = np.interp(H_int, H_model, P_model)
        T_int = np.interp(H_int, H_model, T_model)
        PARA_int = np.interp(H_int, H_model, PARA_model)
        K = 1.37947E-23
        dU_dS_int = P_int / (K * T_int)
        U_layer[ilayer] = simps(dU_dS_int, S_int)
        H_layer[ilayer] = simps(H_int * dU_dS_int, S_int) / U_layer[ilayer]
        P_layer[ilayer] = simps(P_int * dU_dS_int, S_int) / U_layer[ilayer]
        T_layer[ilayer] = simps(T_int * dU_dS_int, S_int) / U_layer[ilayer]
        PARA_layer[ilayer] = simps(PARA_int * dU_dS_int, S_int) / U_layer[ilayer]

        for J in range(Ngas):
            VMR_int = np.interp(H_int, H_model, VMR_model[:, J])
            VMR_layer[ilayer, J] = simps(P_int * VMR_int * dU_dS_int, S_int) / (U_layer[ilayer] * P_layer[ilayer])
        for J in range(Nmodes):
            A_int = np.interp(H_int, H_model, A_model[:, J])
            A_layer[ilayer, J] = simps(A_int * dU_dS_int, S_int)

    U_layer /= scale
    return H_layer, P_layer, T_layer, VMR_layer, U_layer, A_layer, PARA_layer, dH, scale

def calc_layer(planet_radius, H_model, P_model, T_model, VMR_model, A_model, PARA_model, ID, NLAYER,
               path_angle, H_0=0.0, layer_type=1, custom_path_angle=0.0,
               custom_H_base=None, custom_P_base=None):
    """
    Top level routine that calculates the layer properties from an atmospheric model.
    """
    H_base, P_base = split(H_model, P_model, NLAYER, layer_type=layer_type, H_0=H_0, 
                           planet_radius=planet_radius, custom_path_angle=custom_path_angle, 
                           custom_H_base=custom_H_base, custom_P_base=custom_P_base)
    
    H_layer, P_layer, T_layer, VMR_layer, U_layer, A_layer, PARA_layer, dH, scale = average(
        planet_radius, H_model, P_model, T_model, VMR_model, A_model, PARA_model, ID, 
        H_base, path_angle=path_angle, H_0=H_0)
    
    return H_layer, P_layer, T_layer, VMR_layer, U_layer, A_layer, PARA_layer, dH, scale
