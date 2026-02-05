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

import unittest
import numpy as np
from triqs_hartree_fock import ImpuritySolver
from triqs_hartree_fock.utils import flatten
from triqs.gf import *
from triqs.operators import *
from h5 import HDFArchive
from triqs.lattice.tight_binding import TBLattice
from triqs.sumk import *
from triqs.lattice import *


def set_G0_iw_from_Gloc(solver, Gloc, gf_struct):
    """Set solver's G0_iw by evaluating Weiss field at DLR frequency points."""
    for bl_name, _ in gf_struct:
        for iw in solver.G0_iw[bl_name].mesh:
            Gloc_iw = Gloc[bl_name](iw.value)
            Sigma_bl = solver.Sigma_HF[bl_name]
            G0_iw = np.linalg.inv(np.linalg.inv(Gloc_iw) + Sigma_bl)
            solver.G0_iw[bl_name][iw] = G0_iw


class test_impurity_solver(unittest.TestCase):

    def test_dc(self):

        t = 1
        tp = 0.1
        hop = {(1, 0):  [[t]],
               (-1, 0):  [[t]],
               (0, 1):  [[t]],
               (0, -1):  [[t]],
               (1, 1):  [[tp]],
               (-1, -1):  [[tp]],
               (1, -1):  [[tp]],
               (-1, 1):  [[tp]]}

        TBL = TBLattice(units=[(1, 0, 0), (0, 1, 0)], hoppings=hop, orbital_positions=[(0., 0., 0.)]*1)
        nk = 10
        beta = 40
        h_int = 3*n('up', 0)*n('down', 0)
        gf_struct = [('up', 1), ('down', 1)]

        SK = SumkDiscreteFromLattice(lattice=TBL, n_points=nk)
        w_max, eps = 10.0, 1e-10
        # External Sigma/Gloc use MeshImFreq for compatibility with SumkDiscreteFromLattice
        sigma = GfImFreq(beta=beta, n_points=1025, target_shape=[1, 1])
        Sigma = BlockGf(name_list=['up', 'down'], block_list=(sigma, sigma), make_copies=True)
        Gloc = Sigma.copy()
        mu = 0
        # ImpuritySolver uses MeshDLRImFreq internally
        S = ImpuritySolver(gf_struct=gf_struct, beta=beta, w_max=w_max, eps=eps, dc=True, dc_U=2.0, dc_J=0.2, dc_type='cFLL')

        converged = False
        while not converged:
            for name, _ in gf_struct:
                Sigma[name] << S.Sigma_HF[name]
            # mu, density = dichotomy(lambda mu: SK(mu=mu, Sigma=Sigma).total_density().real, mu, density_required,
            #                         1e-5, .5, max_loops = 100, x_name="chemical potential", y_name="density", verbosity=3)
            Gloc << SK(mu=mu, Sigma=Sigma)
            set_G0_iw_from_Gloc(S, Gloc, gf_struct)
            Sigma_old = S.Sigma_HF.copy()
            S.solve(h_int=h_int, one_shot=False, tol=1e-4)
            if np.allclose(flatten(Sigma_old), flatten(S.Sigma_HF), rtol=0, atol=1e-6):
                converged = True


if __name__ == '__main__':
    unittest.main()
