#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Calculate mean molecular weight
"""
from nemesispy_scatter.common.info_mol import mol_info
from nemesispy_scatter.common.constants import AMU
def calc_mmw(ID, VMR, ISO=[]):
    """
    Calculate mean molecular weight in kg given a list of molecule IDs and
    a list of their respective volume mixing ratios.

    :Parameters:
    ------------
        ID : ndarray or list
            A list of Radtran gas identifiers.
        VMR : ndarray or list
            A list of VMRs corresponding to the gases in ID.
        ISO : ndarray or list
            If ISO=[], assume terrestrial relative isotopic abundance for all gases.
            Otherwise, if ISO[i]=0, then use terrestrial relative isotopic abundance
            for the ith gas. To specify particular isotopologue, input the
            corresponding Radtran isotopologue identifiers.

    :Returns:
    ---------
        mmw : real
            Mean molecular weight.
            Unit: kg

    Notes
    -----
    Cf mol_id.py and mol_info.py.
    """
    mmw = 0
    for gas_index, gas_id in enumerate(ID):
        gas_info = mol_info[f'{gas_id}']['isotope']

        if len(ISO) == 0 or ISO[gas_index] == 0:
            for isotope, isotope_info in gas_info.items():
                mmw += isotope_info['mass'] * isotope_info['abun'] * VMR[gas_index]
        else:
            # Assuming 'isotope' variable is defined elsewhere when ISO[gas_index] != 0
            isotope_info = gas_info[f'{ISO[gas_index]}']
            mmw += isotope_info['mass'] * VMR[gas_index]
        
    mmw *= AMU # keep in SI unit
    return mmw
