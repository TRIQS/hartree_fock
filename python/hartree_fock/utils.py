# Copyright (c) 2022 Simons Foundation
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You may obtain a copy of the License at
#     https:#www.gnu.org/licenses/gpl-3.0.txt
#
# Authors: Jonathan Karp, Alexander Hampel, Nils Wentzell, Hugo U. R. Strand, Olivier Parcollet

import numpy as np
import triqs.utility.mpi as mpi


def flatten(Sigma_HF, real=False):
    """
    Flatten a dictionary of 2D Numpy arrays into a 1D Numpy array.

    Parameters
    ----------
    Sigma_HF : dictionary of 2D arrays

    real : optional, bool
        True if the Numpy arrays have a real dtype. Default is False.

    """
    if real:
        return np.array([Sig_bl.flatten() for bl, Sig_bl in Sigma_HF.items()]).flatten()
    else:
        return np.array([Sig_bl.flatten().view(float) for bl, Sig_bl in Sigma_HF.items()]).flatten()


def unflatten(Sigma_HF_flat, gf_struct, real=False):
    """
    Unflatten a 1D Numpy array back to a dictionary of 2D Numpy arrays, based on the structure in gf_struct.

    Parameters
    ----------
    Sigma_HF_flat : 1D Numpy array

    gf_struct: list of pairs [ (str,int), ...]
        Structure of the Green's functions. Used to
        construct the dictionary of 2D arrays.

    real : optional, bool
        True if the Numpy array has a real dtype. Default is False.
    """

    offset = 0
    Sigma_HF = {}
    for bl, bl_size in gf_struct:
        if real:
            Sigma_HF[bl] = Sigma_HF_flat[list(range(offset, offset + bl_size**2))].reshape((bl_size, bl_size))
            offset = offset + bl_size**2
        else:
            Sigma_HF[bl] = Sigma_HF_flat[list(range(offset, offset + 2*bl_size**2))].view(complex).reshape((bl_size, bl_size))
            offset = offset + 2*bl_size**2
    return Sigma_HF


def fermi(e, beta):
    """
    Numerically stable version of the Fermi function

    Parameters
    ----------
    e : float or ndarray
        Energy minus chemical potential

    beta: float
        Inverse temperature

    """
    return np.exp(-beta * e * (e > 0))/(1 + np.exp(-beta*np.abs(e)))

def compute_DC_from_density(N_up, N_down, U, J, n_orbitals=5,  method='cFLL', spin_channel=None):
    """
    To integrate with SUMK
    Returns a float for the DC correction  


    Parameters
    ----------
    N_up : float 
        Spin up total density
    
    N_down : float 
        Spin down total density

    U : float 
        U value

    J : float 
        J value

    n_orbitals : int, default = 5
        Total number of orbitals
    
    spin_channel : string, default = None
        For which spin channel you are computing the DC correction for, possibilities :
        -   None: no spin resolved
        -   up: up channel
        -   down: down channel
    
    method : string, default = 'cFLL' 
        possibilities:
        -    cFLL: DC potential from Ryee for spin unpolarized DFT: (DOI: 10.1038/s41598-018-27731-4)
        -    sFLL: TO IMPLEMENT, same as above for spin polarized DFT
        -    cAMF: TO IMPLEMENT
        -    sAMF: TO IMPLEMENT
        -    cHeld: unpolarized Held's formula as reported in (DOI: 10.1103/PhysRevResearch.2.03308)
        -    sHeld: polarized Held's formula as reported in (DOI: 10.1103/PhysRevResearch.2.03308)
    """
    N_tot = N_up + N_down

    match method:
        case 'cFLL':
            DC_val = U * (N_tot-0.5) - J *(N_tot*0.5-0.5)

        case 'sFLL':
            if spin_channel == 'up':
                DC_val = U * (N_tot-0.5) - J *(N_up-0.5)
            elif spin_channel == 'down':
                DC_val = U * (N_tot-0.5) - J *(N_down-0.5)
            else:
                raise ValueError(f"spin_channel set to {spin_channel}, please select 'up' or 'down'")
        
        case 'cHeld':
            U_mean = U * (n_orbitals-1)*(U-2*J)+(n_orbitals-1)*(U-3*J)/(2*n_orbitals-1)
            DC_val = U_mean * (N_tot-0.5)

    mpi.report(f"DC computed using the {method} method for a value of {DC_val:.6f} eV")
    if 'Held' in method:
        mpi.report(f"Held method for {n_orbitals} orbitals, computed U_mean={U_mean:.6f} eV")

    return DC_val


def logo():
    logo = """
╔╦╗╦═╗╦╔═╗ ╔═╗  ┬ ┬┌─┐
 ║ ╠╦╝║║═╬╗╚═╗  ├─┤├┤
 ╩ ╩╚═╩╚═╝╚╚═╝  ┴ ┴└
TRIQS: Hartree-Fock solver
"""
    return logo

