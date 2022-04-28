#!/usr/bin/env python

import unittest
import numpy as np
from hartree_fock.solver import LatticeSolver
from triqs.gf import *
from triqs.lattice import *
from triqs.operators import *
from hartree_fock.solver import flatten, unflatten
from h5 import HDFArchive

class test_lattice_solver(unittest.TestCase):

    def test_one_band(self):


        BL = BravaisLattice(units = [(1,0,0) , (0,1,0) , (0,0,1)])
        BZ = BrillouinZone(BL)
        nk=10
        mk = MeshBrZone(BZ, nk)
        ekup = Gf(mesh = mk, target_shape=[1,1])
        ekdn = Gf(mesh = mk, target_shape=[1,1])

        for k in mk:
            ekup[k][0,0] = 1*(np.cos(k[0]) + np.cos(k[1]) + np.cos(k[2]))
            ekdn[k][0,0] = 1*(np.cos(k[0]) + np.cos(k[1]) + np.cos(k[2]))

        h0_k = BlockGf(name_list = ['up', 'down'], block_list = [ekup, ekdn])
        gf_struct = [('up', 1), ('down', 1)]
        h_int = 3*n('up',0)*n('down',0)

        S = LatticeSolver(h0_k=h0_k, gf_struct=gf_struct, beta=40)
        S.solve(h_int=h_int, N_target=1, with_fock=True)

        with HDFArchive('one_band.ref.h5', 'r') as ar:
            Sigma_ref = ar['Sigma']
            mu_ref = ar['mu']
        np.testing.assert_allclose(S.Sigma_HF['up'], Sigma_ref[0,0], rtol=0, atol=1e-10)
        np.testing.assert_allclose(S.Sigma_HF['down'], Sigma_ref[1,1], rtol=0, atol=1e-10)
        np.testing.assert_allclose(S.mu, mu_ref)

        #test with forced symmetry
        def make_spins_equal(Sigma):
            Symmetrized_Sigma = Sigma.copy()
            Symmetrized_Sigma['up'] = 0.5*(Sigma['up']  + Sigma['down'])
            Symmetrized_Sigma['down'] = Symmetrized_Sigma['up']
            return Symmetrized_Sigma

        S = LatticeSolver(h0_k=h0_k, gf_struct=gf_struct, beta=40, symmetries=[make_spins_equal])
        S.solve(h_int=h_int, N_target=1, with_fock=True)
        np.testing.assert_allclose(S.Sigma_HF['up'], Sigma_ref[0,0], rtol=0, atol=1e-10)
        np.testing.assert_allclose(S.Sigma_HF['down'], Sigma_ref[1,1], rtol=0, atol=1e-10)
        np.testing.assert_allclose(S.mu, mu_ref)

        #test forcing Sigma to be real
        S = LatticeSolver(h0_k=h0_k, gf_struct=gf_struct, beta=40, force_real=True)
        S.solve(h_int=h_int, N_target=1, with_fock=True)
        np.testing.assert_allclose(S.Sigma_HF['up'], Sigma_ref[0,0], rtol=0, atol=1e-10)
        np.testing.assert_allclose(S.Sigma_HF['down'], Sigma_ref[1,1], rtol=0, atol=1e-10)
        np.testing.assert_allclose(S.mu, mu_ref)


    def test_multi_band(self):

        BL = BravaisLattice(units = [(1,0,0) , (0,1,0) , (0,0,1)])
        BZ = BrillouinZone(BL)
        nk=10
        mk = MeshBrZone(BZ, nk)
        ekup = Gf(mesh = mk, target_shape=[2,2])
        ekdn = Gf(mesh = mk, target_shape=[2,2])

        for k in mk:
            ekup[k][0,0] = 2*np.cos(k[0]) + np.cos(k[1]) + np.cos(k[2])
            ekup[k][1,1] = np.cos(k[0]) + np.cos(k[1]) + np.cos(k[2])
            ekup[k][0,1] = 0.1*np.cos(k[0])
            ekup[k][1,0] = 0.1*np.cos(k[0])
            ekdn[k] = (np.cos(k[0]) + np.cos(k[1]) + np.cos(k[2]) - 0.1) * np.identity(2)
            ekdn[k][0,1] = 0.1*np.cos(k[1])
            ekdn[k][1,0] = 0.1*np.cos(k[1])

        h0_k = BlockGf(name_list = ['up', 'down'], block_list = [ekup, ekdn])
        gf_struct = [('up', 2), ('down', 2)]

        h_int = 3*n('up', 0)*n('down', 0) + 3*n('up', 1)*n('down', 1) + 2.5*n('up', 0)*n('down', 1)

        S = LatticeSolver(h0_k=h0_k, gf_struct=gf_struct, beta=40)
        S.solve(h_int=h_int, N_target=2, with_fock=True)
        with HDFArchive('multi_band.ref.h5', 'r') as ar:
            Sigma_ref = ar['Sigma']
            mu_ref = ar['mu']
            rho_ref = ar['rho']
        np.testing.assert_allclose(S.Sigma_HF['up'], Sigma_ref[:2,:2], rtol=0, atol=1e-5)
        np.testing.assert_allclose(S.Sigma_HF['down'], Sigma_ref[2:,2:], rtol=0, atol=1e-5)
        np.testing.assert_allclose(S.rho['up'], rho_ref[:2,:2], rtol=0, atol=1e-5)
        np.testing.assert_allclose(S.rho['down'], rho_ref[2:,2:], rtol=0, atol=1e-5)
        np.testing.assert_allclose(S.mu, mu_ref)

        #test storing to and loading from h5
        with HDFArchive('lattice_results.h5', 'w') as ar:
            ar['solver'] = S
        with HDFArchive('lattice_results.h5', 'r') as ar:
            S = ar['solver']

if __name__ == '__main__':
    unittest.main()
