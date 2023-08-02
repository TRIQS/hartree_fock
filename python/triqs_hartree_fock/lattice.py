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

import copy
import numpy as np
from scipy.optimize import root, brentq
from triqs.gf import *
import triqs.utility.mpi as mpi
from h5.formats import register_class
from .utils import *


class LatticeSolver(object):

    """ Hartree-Fock Lattice solver for local interactions.

    Parameters
    ----------

    h0_k : TRIQS BlockGF on a Brillouin zone mesh
        Single-particle dispersion.

    gf_struct : list of pairs [ (str,int), ...]
        Structure of the Green's functions. It must be a
        list of pairs, each containing the name of the
        Green's function block as a string and the size of that block.
        For example: ``[ ('up', 3), ('down', 3) ]``.

    beta : float
        inverse temperature

    symmeties : optional, list of functions
        symmetry functions acting on self energy at each consistent step

    force_real : optional, bool
        True if the self energy is forced to be real

    """

    def __init__(self, h0_k, gf_struct, beta, symmetries=[], force_real=False):

        self.h0_k = h0_k.copy()
        self.h0_k_MF = h0_k.copy()
        self.n_k = len(self.h0_k.mesh)

        self.gf_struct = gf_struct
        self.beta = beta
        self.symmetries = symmetries
        self.force_real = force_real

        if self.force_real:
            self.Sigma_HF = {bl: np.zeros((bl_size, bl_size)) for bl, bl_size in gf_struct}
        else:
            self.Sigma_HF = {bl: np.zeros((bl_size, bl_size), dtype=complex) for bl, bl_size in gf_struct}
        self.rho = {bl: np.zeros((bl_size, bl_size)) for bl, bl_size in gf_struct}

        self.git_hash = "@PROJECT_GIT_HASH@"

    def solve(self, h_int, N_target=None, mu=None, with_fock=True, one_shot=False, method='broyden1', tol=None):
        """ Solve for the Hartree Fock self energy using a root finder method.
        The self energy is stored in the ``Sigma_HF`` object of the LatticeSolver instance.
        If a fixed target density ``N_target`` is given, then the chemical potential is calculated
        and stored in the ``mu`` object of the LatticeSolver instance.

        Parameters
        ----------

        h_int : TRIQS Operator instance
            Local interaction Hamiltonian

        N_target : optional, float
            target density per site. Can only be provided if mu is not provided

        mu: optional, float
            chemical potential. Can only be provided if N_target is not provided. Default is 0 if N_target is not provided

        with_fock : optional, bool
            True if the fock terms are included in the self energy. Default is True

        one_shot : optional, bool
            True if the calcualtion is just one shot and not self consistent. Default is False

        method : optional, string
            method for root finder. Only used if one_shot=False, see scipy.optimize.root
            for options. Default : 'broyden1'

        tol: optional, float
            Convergence tolerance to pass to root finder

        """
        self.last_solve_params = {key: val for key, val in locals().items() if key != 'self'}
        mpi.report(logo())
        if mu is not None and N_target is not None:
            raise ValueError('Only provide either mu or N_target, not both')

        if not N_target is None:
            self.fixed = 'density'
            self.N_target = N_target
        else:
            self.fixed = 'mu'
            if not mu is None:
                self.mu = mu
            else:
                self.mu = 0

        if self.fixed == 'density':
            mpi.report('Running Lattice Solver at fixed density of %.4f' % self.N_target)
        else:
            mpi.report('Running Lattice Solver at fixed chemical potential of %.4f' % self.mu)
        mpi.report('beta = %.4f' % self.beta)
        mpi.report('h_int =', h_int)
        if one_shot:
            mpi.report('mode: one shot')
        else:
            mpi.report('mode: self-consistent')
        mpi.report('Including Fock terms:', with_fock)

        # function to pass to root finder
        def target_function(Sigma_HF_flat):
            self.update_mean_field_dispersion(unflatten(Sigma_HF_flat, self.gf_struct, real=self.force_real))
            if self.fixed == 'density':
                self.update_mu(self.N_target)
            rho = self.update_rho()
            if self.force_real:
                Sigma_HF = {bl: np.zeros((bl_size, bl_size)) for bl, bl_size in self.gf_struct}
            else:
                Sigma_HF = {bl: np.zeros((bl_size, bl_size), dtype=complex) for bl, bl_size in self.gf_struct}
            for term, coef in h_int:
                bl1, u1 = term[0][1]
                bl2, u2 = term[3][1]
                bl3, u3 = term[1][1]
                bl4, u4 = term[2][1]

                assert(bl1 == bl2 and bl3 == bl4)
                Sigma_HF[bl1][u2, u1] += coef * rho[bl3][u4, u3]
                Sigma_HF[bl3][u4, u3] += coef * rho[bl1][u2, u1]

                if bl1 == bl3 and with_fock:
                    Sigma_HF[bl1][u4, u1] -= coef * rho[bl3][u2, u3]
                    Sigma_HF[bl3][u2, u3] -= coef * rho[bl1][u4, u1]
            for function in self.symmetries:
                Sigma_HF = function(Sigma_HF)
            if one_shot:
                return Sigma_HF
            return Sigma_HF_flat - flatten(Sigma_HF, real=self.force_real)

        Sigma_HF_init = self.Sigma_HF

        if one_shot:
            self.Sigma_HF = target_function(Sigma_HF_init)

        else:  # self consistent Hartree-Fock
            if tol is None:
                root_finder = root(target_function, flatten(Sigma_HF_init), method=method)
            else:
                root_finder = root(target_function, flatten(Sigma_HF_init), method=method, tol=tol)
            if root_finder['success']:
                mpi.report('Self Consistent Hartree-Fock converged successfully')
                self.Sigma_HF = unflatten(root_finder['x'], self.gf_struct, self.force_real)
                with np.printoptions(suppress=True, precision=4):
                    for name, bl in self.Sigma_HF.items():
                        mpi.report('Sigma_HF[\'%s\']:' % name)
                        mpi.report(bl)
                if self.fixed == 'density':
                    mpi.report('mu = %.4f' % self.mu)

            else:
                mpi.report('Hartree-Fock solver did not converge successfully.')
                mpi.report(root_finder['message'])

    def update_mean_field_dispersion(self, Sigma_HF):
        for bl, size in self.gf_struct:
            self.h0_k_MF[bl].data[:] = self.h0_k[bl].data + Sigma_HF[bl][None, ...]

    def update_rho(self):

        for bl, size in self.gf_struct:
            e, V = np.linalg.eigh(self.h0_k_MF[bl].data)
            e -= self.mu

            # density matrix = Sum fermi_function*|psi><psi|
            dm = np.einsum('kab,kb,kcb->ac', V, fermi(e, self.beta), V.conj())/self.n_k
            if self.force_real:
                max_imag = dm.imag.max()
                if max_imag > 1e-10:
                    mpi.report('Warning! Discarding imaginary part of density matrix. Largest imaginary part: %f' % max_imag)
                dm = dm.real
            self.rho[bl] = dm

        return self.rho

    def update_mu(self, N_target):

        energies = {}
        e_min = np.inf
        e_max = -np.inf
        for bl, size in self.gf_struct:
            energies[bl] = np.linalg.eigvalsh(self.h0_k_MF[bl].data)
            bl_min = energies[bl].min()
            bl_max = energies[bl].max()
            if bl_min < e_min:
                e_min = bl_min
            if bl_max > e_max:
                e_max = bl_max

        def target_function(mu):
            n = 0
            for bl, size in self.gf_struct:
                n += np.sum(fermi(energies[bl] - mu, self.beta)) / self.n_k
            return n - N_target
        mu = brentq(target_function, e_min, e_max)
        self.mu = mu
        return mu

    def __reduce_to_dict__(self):
        store_dict = {'h0_k': self.h0_k, 'h0_k_MF': self.h0_k_MF, 'n_k': self.n_k,
                      'gf_struct': self.gf_struct, 'beta': self.beta,
                      'symmetries': self.symmetries, 'force_real': self.force_real,
                      'Sigma_HF': self.Sigma_HF, 'rho': self.rho, 'git_hash': self.git_hash}
        if hasattr(self, 'mu'):
            store_dict['mu'] = self.mu
        if hasattr(self, 'last_solve_params'):
            params_copy = copy.deepcopy(self.last_solve_params)
            if params_copy['mu'] is None:
                params_copy['mu'] = 'none'
            if params_copy['tol'] is None:
                params_copy['tol'] = 'none'
            store_dict['last_solve_params'] = params_copy

        return store_dict

    @classmethod
    def __factory_from_dict__(cls, name, D):

        instance = cls(D['h0_k'], D['gf_struct'], D['beta'], D['symmetries'], D['force_real'])

        instance.h0_k_MF = D['h0_k_MF']
        instance.n_k = D['n_k']
        instance.Sigma_HF = D['Sigma_HF']
        instance.rho = D['rho']
        instance.git_hash = D['git_hash']
        if 'mu' in D:
            instance.mu = D['mu']
        if 'last_solve_params' in D:
            params = copy.deepcopy(D['last_solve_params'])
            if params['mu'] == 'none':
                params['mu'] = None
            if params['tol'] == 'none':
                params['tol'] = None
            instance.last_solve_params = params
        return instance


register_class(LatticeSolver)
