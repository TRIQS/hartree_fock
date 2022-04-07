#!/usr/bin/env python

import unittest
import numpy as np
from hartree_fock.solver import LatticeSolver
from triqs.gf import *
from triqs.lattice import *
from triqs.operators import *
from hartree_fock.solver import flatten, unflatten
class test_lattice_solver(unittest.TestCase):

    def test_run(self):

        # BL = BravaisLattice(units = [(1,0,0) , (0,1,0) , (0,0,1)])
        # BZ = BrillouinZone(BL)
        # nk=10
        # mk = MeshBrZone(BZ, nk)
        # ekup = Gf(mesh = mk, target_shape=[2,2])
        # ekdn = Gf(mesh = mk, target_shape=[2,2])

        # for k in mk:
        #     ekup[k][0,0] = 2*np.cos(k[0]) + np.cos(k[1]) + np.cos(k[2])
        #     ekup[k][1,1] = np.cos(k[0]) + np.cos(k[1]) + np.cos(k[2])
        #     ekup[k][0,1] = 0.1*np.cos(k[0])
        #     ekup[k][1,0] = 0.1*np.cos(k[0])
        #     ekdn[k] = (np.cos(k[0]) + np.cos(k[1]) + np.cos(k[2]) - 10) * np.identity(2)
        #     ekdn[k][0,1] = 0.1*np.cos(k[1])
        #     ekdn[k][1,0] = 0.1*np.cos(k[1])

        # e_k = BlockGf(name_list = ['up', 'down'], block_list = [ekup, ekdn])
        # gf_struct = [('up', 2), ('down', 2)]

        # h_int = 3*n('up', 0)*n('down', 0) + 3*n('up', 1)*n('down', 1) + 2.5*n('up', 0)*n('down', 1)

        # S = LatticeSolver(e_k=e_k, h_int=h_int, gf_struct=gf_struct, beta=40)
        # S.solve(N_target=2, with_fock=True)
        # print('Sigma_up = ', S.Sigma_HF['up'])
        # print('Sigma_down = ', S.Sigma_HF['down'])
        # print('mu = ', S.mu)
        # print('rho = ', S.rho)


        BL = BravaisLattice(units = [(1,0,0) , (0,1,0) , (0,0,1)])
        BZ = BrillouinZone(BL)
        nk=10
        mk = MeshBrZone(BZ, nk)
        ekup = Gf(mesh = mk, target_shape=[1,1])
        ekdn = Gf(mesh = mk, target_shape=[1,1])

        for k in mk:
            ekup[k][0,0] = 1*(np.cos(k[0]) + np.cos(k[1]) + np.cos(k[2]))
            ekdn[k][0,0] = 1*(np.cos(k[0]) + np.cos(k[1]) + np.cos(k[2]))

        e_k = BlockGf(name_list = ['up', 'down'], block_list = [ekup, ekdn])
        gf_struct = [('up', 1), ('down', 1)]
        h_int = 3*n('up',0)*n('down',0)

        S = LatticeSolver(e_k=e_k, h_int=h_int, gf_struct=gf_struct, beta=40)
        S.solve(N_target=1, with_fock=True)
        print('Sigma_up = ', S.Sigma_HF['up'])
        print('Sigma_down = ', S.Sigma_HF['down'])
        print('mu = ', S.mu)
        print('rho = ', S.rho)

if __name__ == '__main__':
    unittest.main()
