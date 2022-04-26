#!/usr/bin/env python

import unittest
import numpy as np
from hartree_fock.solver import ImpuritySolver, LatticeSolver, flatten
from triqs.gf import *
from triqs.operators import *
from h5 import HDFArchive
from triqs.lattice.tight_binding import TBLattice
from triqs.sumk import *
from triqs.lattice import *
from triqs.utility.dichotomy import dichotomy

class test_impurity_solver(unittest.TestCase):

    #test that lattice and impurity solvers agree
    def test_agreement(self):

        t = 1
        tp = 0.1
        hop= {  (1,0)  :  [[ t]],       
                (-1,0) :  [[ t]],     
                (0,1)  :  [[ t]],
                (0,-1) :  [[ t]],
                (1,1)  :  [[ tp]],
                (-1,-1):  [[ tp]],
                (1,-1) :  [[ tp]],
                (-1,1) :  [[ tp]]}

        TBL = TBLattice(units = [(1, 0, 0) , (0, 1, 0)], hoppings = hop, orbital_positions= [(0., 0., 0.)]*1)
        nk = 10
        beta=40
        h_int = 3*n('up', 0)*n('down', 0)
        gf_struct = [('up', 1), ('down', 1)]

        SK = SumkDiscreteFromLattice(lattice=TBL, n_points=nk)
        sigma = GfImFreq(beta=beta, n_points=1025, target_shape=[1,1])
        Sigma = BlockGf(name_list=['up', 'down'], block_list=(sigma,sigma), make_copies=True)
        Gloc = Sigma.copy()
        # density_required = 1
        mu = 0
        S = ImpuritySolver(h_int=h_int, gf_struct=gf_struct, beta=beta, n_iw=1025)

        converged = False
        while not converged:
            for name, bl in gf_struct:
                Sigma[name] << S.Sigma_HF[name]
            # mu, density = dichotomy(lambda mu: SK(mu=mu, Sigma=Sigma).total_density().real, mu, density_required,
            #                         1e-5, .5, max_loops = 100, x_name="chemical potential", y_name="density", verbosity=3)
            Gloc << SK(mu=mu, Sigma=Sigma)
            S.G0_iw << inverse(inverse(Gloc) + Sigma)
            Sigma_old = S.Sigma_HF.copy()
            S.solve()
            if np.allclose(flatten(Sigma_old), flatten(S.Sigma_HF), rtol=0, atol=1e-6):
                converged = True
        Sigma_imp = S.Sigma_HF
        mu_imp = mu

        #test storing to and loading from h5
        with HDFArchive('impurity_results.h5', 'w') as ar:
            ar['solver'] = S
        with HDFArchive('impurity_results.h5', 'r') as ar:
            S = ar['solver']

        BL = BravaisLattice(units = [(1,0,0) , (0,1,0)])
        BZ = BrillouinZone(BL)
        mk = MeshBrZone(BZ, nk)
        ekup = Gf(mesh=mk, target_shape=[1,1])
        ekdn = Gf(mesh=mk, target_shape=[1,1])
        ekup << TBL.fourier(mk)
        ekdn << TBL.fourier(mk) 
        h0_k = BlockGf(name_list=['up', 'down'], block_list=(ekup, ekdn))
        S = LatticeSolver(h0_k=h0_k, h_int=h_int, gf_struct=gf_struct, beta=beta)
        # S.solve(N_target=1)
        S.solve(mu=0)
        np.testing.assert_allclose(flatten(S.Sigma_HF), flatten(Sigma_imp), rtol=0, atol=1e-4)
        # np.testing.assert_allclose(S.mu, mu_imp, rtol=0, atol=1e-4)

if __name__ == '__main__':
    unittest.main()
