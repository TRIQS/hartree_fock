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

