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


def logo():
    logo = """
╔╦╗╦═╗╦╔═╗ ╔═╗  ┬ ┬┌─┐
 ║ ╠╦╝║║═╬╗╚═╗  ├─┤├┤
 ╩ ╩╚═╩╚═╝╚╚═╝  ┴ ┴└
TRIQS: Hartree-Fock solver
"""
    return logo

def compute_DC_from_density(N_tot, U, J, N_spin=None,  n_orbitals=5,  method='sFLL'):
    """
    Computes the double counting correction using various methods.
    For FLL and AMF DC the notations and equations from doi.org/10.1038/s41598-018-27731-4
    are used, whereas for the Held DC the definitions from doi.org/10.1080/00018730701619647 are used.

    TODO: this is a copy of triqs_dft_tools.util.compute_DC_from_density to remove dependency on dfttools
    Parameters
    ----------
    N_tot : float 
        Total density of the impurity
    N_spin : float , default = None
        Spin density, defaults to N_tot*0.5 if not specified
    U : float 
        U value
    J : float 
        J value
    n_orbitals : int, default = 5
        Total number of orbitals
    method : string, default = 'cFLL' 
        possibilities:
        -    cFLL: DC potential from Ryee for spin unpolarized DFT: (DOI: 10.1038/s41598-018-27731-4)
        -    sFLL: same as above for spin polarized DFT
        -    cAMF: around mean field
        -    sAMF: spin polarized around mean field
        -    cHeld: unpolarized Held's formula as reported in (DOI: 10.1103/PhysRevResearch.2.03308)
        -    sHeld: NOT IMPLEMENTED
    
    Returns
    -------
    List of floats:
        -   DC_val: double counting potential
        -   E_val: double counting energy


    todo: 
        - See whether to move this to TRIQS directly instead of dft_tools
        - allow as input full density matrix to allow orbital dependent DC
    """
    if N_spin is not None:
        N_spin2 = N_tot-N_spin
        Mag = N_spin - N_spin2
    L_orbit = (n_orbitals-1)/2

    if method == 'cFLL':
        E_val = 0.5 * U * N_tot * (N_tot-1) - 0.5 * J * N_tot * (N_tot*0.5-1)
        DC_val = U * (N_tot-0.5) - J * (N_tot*0.5-0.5)

    elif method == 'sFLL':
        assert N_spin is not None, "Spin density not given"
        E_val = 0.5 * U * N_tot * (N_tot-1) - 0.5 * J * N_tot * (N_tot*0.5-1) - 0.25 * J * Mag**2
        DC_val = U * (N_tot-0.5) - J * (N_spin-0.5)

    elif method == 'cAMF':
        E_val = +0.5 * U * N_tot ** 2
        E_val -= 0.25*(U+2*L_orbit*J)/(2*L_orbit+1)*N_tot**2
        DC_val = U * N_tot - 0.5*(U+2*L_orbit*J)/(2*L_orbit+1)*N_tot

    elif method == 'sAMF':
        assert N_spin is not None, "Spin density not given"
        E_val = 0.5 * U * N_tot ** 2
        E_val -= 0.25*(U+2*L_orbit*J)/(2*L_orbit+1)*N_tot**2
        E_val -= 0.25*(U+2*L_orbit*J)/(2*L_orbit+1)*Mag**2
        DC_val = U * N_tot - (U+2*L_orbit*J)/(2*L_orbit+1)*N_spin

    elif method == 'cHeld':
        # Valid for a Kanamori-type Hamiltonian where U'=U-2J
        U_mean = (U + (n_orbitals-1)*(U-2*J)+(n_orbitals-1)*(U-3*J))/(2*n_orbitals-1)
        E_val = 0.5 * U_mean * N_tot * (N_tot - 1)
        DC_val = U_mean * (N_tot-0.5)

    elif method == 'sHeld':
        raise ValueError(f"Method sHeld not yet implemented")

    else:
        raise ValueError(f"DC type {method} not supported")

    mpi.report(f"DC potential computed using the {method} method, V_DC = {DC_val:.6f} eV")
    mpi.report(f"E_DC using the {method} method, E_DC = {E_val:.6f} eV")
    if 'Held' in method:
        mpi.report(f"Held method for {n_orbitals} orbitals, computed U_mean={U_mean:.6f} eV")

    return DC_val, E_val

